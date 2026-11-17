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
"""

# The non-deterministic Cake dense GEMMs (the public module ``flashinfer.cake_dense_gemm_nondet_train``, backend
# ``flashinfer.experimental.cake.cake_dense_gemm_nondet_train``): the routes of ``cake_backend.ROUTES``, PyTorch's
# deterministic-algorithms alert before an order-varying launch, the K1 and K2 planners (the per-architecture
# planners, the rule tables, the token-count guard, the stream-K partition), the registry and file names of this
# package's programs and its workspace kinds.  The public names (the mode word, no mode keyword, the former names and
# paths gone, the lazy import) are checked by ``test_cake_dense_gemm_nondet_train_names.py``.  The CPU tests launch
# nothing.  The GPU tests (``test_gpu_*``) need a device the deterministic package registers programs for; the ones of
# this package's programs force the split count (they do not depend on what the planner takes on the device), assert
# by kernel name that the program ran, and skip only on a checkout that registers no program of this package for the
# device.

import ast
import contextlib
import dataclasses
import functools
import gc
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

# Backend suite: the packages are named through ``cake_package``; nothing else of flashinfer is imported here.
package = cake_package("cake_dense_gemm_nondet_train")
cake_backend = cake_package("cake_dense_gemm_nondet_train", "cake_backend")
cake_jit = cake_package("cake_dense_gemm_nondet_train", "cake_jit")
det_backend = cake_package("cake_dense_gemm_det_train", "cake_backend")
det_jit = cake_package("cake_dense_gemm_det_train", "cake_jit")
# A vendored run imports both packages under other names: bind the deterministic package explicitly (in a
# FlashInfer tree this names the sibling the package resolves by itself).
cake_backend.use_deterministic_backend(det_backend)

# Accuracy gates (the deterministic package's budgets, which this module keeps).
BF16_ATOL = BF16_RTOL = 1e-2
F32_REL_FRO_MAX = 5e-5
F32_MAX_ABS_OVER_REF_ABSMAX = 1e-3
ROUTER_REL_FRO_MAX = 2e-6
RERUNS = 20
SEED = 20261031

# A generated name (program, kernel, template, file) of this package never carries the token ``det``.
DET_TOKEN = re.compile(r"(?:^|[^a-z])det(?:[^a-z]|$)", re.IGNORECASE)


class Recorder:
    """A stand-in entry point: records its calls and returns a sentinel."""

    def __init__(self):
        self.calls = []
        self.result = object()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@contextlib.contextmanager
def _no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture
def torch_determinism():
    """Restore PyTorch's deterministic-algorithms setting after the test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _det_name(name: str) -> str:
    """The deterministic package's entry point of the operation of this package's entry point ``name``."""
    return name.replace("cake_dense_gemm_nondet_train", "cake_dense_gemm_det_train")


def _patch_det(monkeypatch, name: str) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(det_backend, _det_name(name), recorder)
    return recorder


def _k1_views(layout: str):
    """CPU views ``(A, B, out)`` of one K1 layout class (``kk``, ``kn``, ``nn``): ``A[M, K]``, ``B[K, N]``."""
    M, N, K = 8, 16, 32
    A = torch.zeros(M, K) if layout[0] == "k" else torch.zeros(K, M).t()
    B = torch.zeros(N, K).t() if layout[1] == "k" else torch.zeros(K, N)
    return A, B, torch.zeros(M, N)


# ---------------------------------------------------------------------------
# Package and module surface (CPU)
# ---------------------------------------------------------------------------


def test_package_init_imports_nothing():
    init = Path(package.__file__)
    tree = ast.parse(init.read_text())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert package.__all__ == ["cake_backend", "cake_build", "cake_jit"]
    assert package.cake_backend is cake_backend and package.cake_jit is cake_jit
    with pytest.raises(AttributeError):
        _ = package.no_such_module


def test_the_package_never_imports_flashinfer_by_name():
    """The package's own modules reach the deterministic package as a sibling; only the pinned loader's JIT fallback
    (``cake_build.py``, shared by every Cake package) imports ``flashinfer.jit``."""
    root = Path(cake_jit.__file__).parent
    for path in sorted(p for p in root.glob("*.py") if p.name != "cake_build.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not [
                    a for a in node.names if a.name.split(".")[0] == "flashinfer"
                ], path.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] != "flashinfer", path.name


def test_cake_build_is_the_pinned_loader():
    ours = Path(cake_jit.__file__).parent / "cake_build.py"
    theirs = Path(det_jit.__file__).parent / "cake_build.py"
    assert ours.read_bytes() == theirs.read_bytes()


def test_the_deterministic_package_resolves_as_a_sibling(monkeypatch):
    """In a FlashInfer tree and in a copy vendored with the deterministic package beside it (both in one parent
    package, or both top-level packages: ``test_cake_dense_gemm_nondet_train_standalone.py``) the deterministic
    package resolves as this package's sibling."""
    monkeypatch.setattr(cake_jit, "_det_package", None)
    assert cake_backend.deterministic_backend() is det_backend
    assert cake_jit.deterministic_module("cake_jit") is det_jit


def test_use_deterministic_backend_binds_a_vendored_copy(monkeypatch):
    monkeypatch.setattr(cake_jit, "_det_package", None)
    fake_pkg, fake = (
        ModuleType("vendored_det_pkg"),
        ModuleType("vendored_det_pkg.cake_backend"),
    )
    for name in cake_backend._DET_ENTRIES:
        setattr(fake, name, Recorder())
    monkeypatch.setitem(sys.modules, "vendored_det_pkg", fake_pkg)
    monkeypatch.setitem(sys.modules, "vendored_det_pkg.cake_backend", fake)
    cake_backend.use_deterministic_backend(fake)
    assert cake_backend.deterministic_backend() is fake
    # a forward layout (routed "det") plans with the bound package's deterministic planner
    A, B, out = _k1_views("kk")
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_plan(A, B, out)
        is fake.cake_dense_gemm_det_train_projection_plan.result
    )
    assert fake.cake_dense_gemm_det_train_projection_plan.calls == [((A, B, out), {})]
    # a knob the caller gives reaches the deterministic planner as given, and only that knob
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_plan(
            A, B, out, sm_count=148, block_n=128
        )
        is fake.cake_dense_gemm_det_train_projection_plan.result
    )
    assert fake.cake_dense_gemm_det_train_projection_plan.calls[1] == (
        (A, B, out),
        {"sm_count": 148, "block_n": 128},
    )
    with pytest.raises(TypeError, match="lacks"):
        cake_backend.use_deterministic_backend(ModuleType("not_a_backend.cake_backend"))


# ---------------------------------------------------------------------------
# Dispatch (CPU)
# ---------------------------------------------------------------------------


def test_router_symbol_names_the_stream_k_templates():
    """``router_symbol`` names this package's stream-K template of the ``kk`` / ``nn_t`` layout classes; ``kn`` (SIMT
    programs only) and ``nk`` have none."""
    symbol = cake_backend.router_symbol
    assert symbol(False, False, False) == "router_fp32_gemm_atomic_kk"
    assert symbol(True, True, True) == "router_fp32_gemm_atomic_nn_t"
    stream_k = {"router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"}
    for arch, table in cake_jit.KERNELS.items():
        assert stream_k <= set(table), arch
    for a_mn, b_mn in ((False, True), (True, False)):
        with pytest.raises(ValueError, match="no stream-K program"):
            cake_backend.router_symbol(a_mn, b_mn, False)


@pytest.mark.parametrize("layout", ["kk", "kn", "nn"])
def test_k1_routes(monkeypatch, layout):
    """The weight-gradient layout reaches this package's split-K path; the forward and
    input-gradient layouts run the deterministic program."""
    det = _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_projection")
    atomic = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k1", atomic)
    A, B, out = _k1_views(layout)
    result = cake_backend.cake_dense_gemm_nondet_train_projection(A, B, out)
    if layout == "nn":
        assert result is atomic.result and not det.calls
        ((args, kwargs),) = atomic.calls
        assert args[0] == "launch" and args[1] is A and args[2] is B and args[3] is out
        assert kwargs["transposed_out"] is False and kwargs["own"] == {}
        assert kwargs["enable_pdl"] is None and kwargs["weights_static"] is False
        assert kwargs["det_call"]() is det.result  # a row the planner does not take
        assert det.calls == [
            ((A, B, out), {"transposed_out": False, "enable_pdl": None})
        ]
        # ... runs the deterministic package's call through this package's binding
        dA, dB, dout, transposed, knobs = kwargs["det_args"]()
        assert (
            dA is A and dB is B and dout is out and transposed is False and knobs == ()
        )
    else:
        # CPU views: the binding hands the call to the deterministic package
        assert result is det.result and not atomic.calls
        assert det.calls == [
            ((A, B, out), {"transposed_out": False, "enable_pdl": None})
        ]


def test_k1_operation_routes(monkeypatch):
    forward, dgrad = (
        _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_projection_forward"),
        _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_projection_dgrad"),
    )
    atomic = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k1", atomic)
    X, W, G = torch.zeros(64, 32), torch.zeros(16, 32), torch.zeros(64, 16)
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W)
        is forward.result
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W) is dgrad.result
    )
    assert not atomic.calls
    out = torch.zeros(16, 32, dtype=torch.bfloat16)
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
        is atomic.result
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(G, X, out)
        is atomic.result
    )
    (launch_args, launch_kwargs), (prepare_args, prepare_kwargs) = atomic.calls
    # the transposed-store launch X.T @ G whatever the deterministic orientation
    assert launch_args[0] == "launch" and prepare_args[0] == "prepare"
    for args, kwargs in ((launch_args, launch_kwargs), (prepare_args, prepare_kwargs)):
        assert args[1].equal(X.t()) and args[2] is G and args[3] is out
        assert kwargs["transposed_out"] is True and kwargs["own"] == {}
    # a row the planner does not take runs the deterministic weight-gradient entry point with its knobs
    wgrad, prepare = (
        _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_projection_wgrad"),
        _patch_det(
            monkeypatch, "cake_dense_gemm_nondet_train_projection_wgrad_prepare"
        ),
    )
    assert launch_kwargs["det_call"]() is wgrad.result
    assert prepare_kwargs["det_call"]() is prepare.result
    assert wgrad.calls == [
        ((G, X, out), {"out_dtype": None, "sk": "auto", "cta_rows": None})
    ]
    assert prepare.calls == [((G, X, out), {"sk": "auto", "cta_rows": None})]
    # ... through this package's binding of the deterministic package's call: its orientation and knobs
    assert (
        launch_kwargs["det_knobs"] == ("auto", None)
        and "det_args" not in prepare_kwargs
    )
    dA, dB, dout, transposed, knobs = launch_kwargs["det_args"]()
    wA, wB, wtransposed = det_backend.wgrad_views(G, X, arch=None, out_f32=False)
    assert dA.equal(wA) and dB.equal(wB) and dout is out and transposed == wtransposed
    assert knobs == (("cta_rows", None), ("sk", "auto"))


@pytest.mark.parametrize("pdl", [None, True])
@pytest.mark.parametrize("op", ["fwd", "dgrad", "kk", "kn"])
def test_k1_deterministic_programs_run_through_the_binding(monkeypatch, op, pdl):
    """K1 calls of this package that run the deterministic program go through this package's binding
    (``_det_k1``): keyed by the entry point's own tensors, with the deterministic package's convenience call as the
    plan (``det_args``) and as the fallback (``det_call``), the entry point's ``enable_pdl`` (and the generic entry's
    ``weights_static``) passed to both."""
    names = {
        "fwd": "cake_dense_gemm_nondet_train_projection_forward",
        "dgrad": "cake_dense_gemm_nondet_train_projection_dgrad",
    }
    det = _patch_det(
        monkeypatch, names.get(op, "cake_dense_gemm_nondet_train_projection")
    )
    binding = Recorder()
    monkeypatch.setattr(cake_backend, "_det_k1", binding)
    X, W, G = torch.zeros(64, 32), torch.zeros(16, 32), torch.zeros(64, 16)
    if op == "fwd":
        call = lambda: cake_backend.cake_dense_gemm_nondet_train_projection_forward(  # noqa: E731
            X, W, enable_pdl=pdl
        )
    elif op == "dgrad":
        call = lambda: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(  # noqa: E731
            G, W, out_dtype=torch.float32, enable_pdl=pdl
        )
    else:
        A, B, out = _k1_views(op)
        call = lambda: cake_backend.cake_dense_gemm_nondet_train_projection(  # noqa: E731
            A, B, out, enable_pdl=pdl, weights_static=bool(pdl)
        )
    assert call() is binding.result and not det.calls
    ((sources, flags, det_args, det_call, enable_pdl, *rest), kwargs) = binding.calls[0]
    # the input gradient names the nvjet program of K1_DGRAD_NVJET_RULES_BY_SM (none for this shape)
    assert kwargs == ({"module": None} if op == "dgrad" else {})
    assert enable_pdl is pdl
    pdl_kw = {} if pdl is None else {"enable_pdl": pdl}
    dA, dB, dout, transposed, knobs = det_args()
    assert dout is sources[-1] and transposed is False and knobs == ()
    if op == "fwd":
        assert sources[0] is X and sources[1] is W and flags == ("fwd",) and not rest
        assert tuple(dout.shape) == (64, 16) and dout.dtype == torch.bfloat16
        assert dA is X and dB.equal(W.t())
        assert det_call() is det.result and det.calls == [((X, W, dout), pdl_kw)]
    elif op == "dgrad":
        assert sources[0] is G and sources[1] is W and flags == ("dgrad",) and not rest
        assert tuple(dout.shape) == (64, 32) and dout.dtype == torch.float32
        assert dA is G and dB is W
        assert det_call() is det.result
        assert det.calls == [((G, W, dout), {"out_dtype": torch.float32, **pdl_kw})]
    else:
        assert all(x is y for x, y in zip(sources, (A, B, out), strict=True))
        assert flags == ("gemm", False) and rest == [bool(pdl)]
        assert dA is A and dB is B
        assert det_call() is det.result
        want = {"transposed_out": False, "enable_pdl": pdl}
        if pdl:
            want["weights_static"] = True
        assert det.calls == [((A, B, out), want)]
    assert len(binding.calls) == 1


def _same_entry_views(views, a_view, split_src, flags) -> None:
    """``views`` (an entry point's own launch views) are ``(a_view, split_src)`` with the plan flags ``flags``."""
    assert _geometry(views[0]) == _geometry(a_view)
    assert views[0].data_ptr() == a_view.data_ptr() and views[1] is split_src
    assert views[2] == flags


def test_k2_routes(monkeypatch):
    """The router weight gradient and input gradient reach this package's K2 path (the input
    gradient without a split count: its SIMT windows, else the deterministic program), and so does the forward with a
    split count or while some router table gives the forward a verdict; without either the forward runs the
    deterministic program through this package's binding (the plan the planner keeps for a row without a verdict).
    Each entry point passes its own tensors as the launch views.  The forward's shipped verdicts (its SIMT windows)
    are cleared first."""
    _without_forward_verdicts(monkeypatch)
    dgrad = _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_router_dgrad")
    gemm = _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_router")
    atomic, splitk = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic)
    monkeypatch.setattr(cake_backend, "_splitk_k2", splitk)
    T, K, N = 64, 32, 16
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W) is splitk.result
    )
    ((args, kwargs),) = splitk.calls
    assert args[0] is X and args[1].equal(W.t()) and args[3] is None and not kwargs
    assert args[5] is None  # enable_pdl: the device's default
    assert tuple(args[2].shape) == (T, N) and args[2].dtype == torch.float32
    _same_entry_views(args[4], X.unsqueeze(0), W, cake_backend._FORWARD_VIEWS)
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, splits=2)
        is atomic.result
    )
    monkeypatch.setitem(cake_backend.ROUTER_RULES, ("sm_103a", "kk", K, N), "S2")
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W) is atomic.result
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, splits=5)
        is atomic.result
    )
    assert cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W) is atomic.result
    assert not dgrad.calls
    calls = atomic.calls
    (
        (fwd2_args, fwd2_kwargs),
        (fwd_args, fwd_kwargs),
        (wg_args, wg_kwargs),
        (dg_args, dg_kwargs),
    ) = calls
    assert fwd2_args[1] is X and fwd2_args[2].equal(W.t())
    assert fwd2_kwargs["splits"] == 2
    assert (
        fwd_args[0] == "launch"
        and fwd_args[1] is X
        and fwd_args[2].equal(W.t())
        and fwd_kwargs["splits"] is None
    )
    for kw in (fwd2_kwargs, fwd_kwargs, wg_kwargs, dg_kwargs):
        assert set(kw) == {"splits", "enable_pdl", "views"} and kw["enable_pdl"] is None
    for kw in (fwd2_kwargs, fwd_kwargs):
        _same_entry_views(kw["views"], X.unsqueeze(0), W, cake_backend._FORWARD_VIEWS)
    assert tuple(fwd_args[3].shape) == (T, N) and fwd_args[3].dtype == torch.float32
    assert wg_args[1].equal(G.t()) and wg_args[2] is X and wg_kwargs["splits"] == 5
    _same_entry_views(wg_kwargs["views"], X.unsqueeze(0), G, cake_backend._WGRAD_VIEWS)
    assert tuple(wg_args[3].shape) == (N, K)
    # the input gradient: no split count (its SIMT windows; the deterministic program's one split elsewhere)
    assert dg_args[0] == "launch" and dg_args[1] is G and dg_args[2] is W
    assert dg_kwargs["splits"] is None
    assert tuple(dg_args[3].shape) == (T, K) and dg_args[3].dtype == torch.float32
    _same_entry_views(dg_kwargs["views"], G.unsqueeze(0), W, cake_backend._DGRAD_VIEWS)
    # the generic entry: every class this package's path (kn: a split count runs the deterministic program there),
    # without entry-point views
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router(
            G, W, torch.zeros(T, K), splits=1
        )
        is atomic.result
    )
    assert (
        atomic.calls[4][1]["splits"] == 1 and atomic.calls[4][1]["enable_pdl"] is None
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router(X, W.t(), torch.zeros(T, N))
        is atomic.result
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router(G.t(), X, torch.zeros(N, K))
        is atomic.result
    )
    assert all(kw["views"] is None for _, kw in atomic.calls[4:])
    assert not gemm.calls and len(splitk.calls) == 1 and len(atomic.calls) == 7


@pytest.mark.parametrize("pdl", [True, False])
def test_k2_routes_pass_enable_pdl(monkeypatch, pdl):
    """``enable_pdl`` reaches every K2 route of this package: this package's binding of the deterministic
    program (the forward without a verdict) and this package's K2 path (the input gradient, the generic ``kn`` call,
    the forward with a split count, the weight gradient).  The forward's shipped verdicts are cleared first."""
    _without_forward_verdicts(monkeypatch)
    atomic, splitk = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic)
    monkeypatch.setattr(cake_backend, "_splitk_k2", splitk)
    T, K, N = 64, 32, 16
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    kw = dict(enable_pdl=pdl)
    cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, **kw)
    cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, **kw)
    cake_backend.cake_dense_gemm_nondet_train_router(
        G, W, torch.zeros(T, K), splits=1, **kw
    )
    cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, splits=2, **kw)
    cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, splits=5, **kw)
    assert [args[5] for args, _ in splitk.calls] == [pdl]
    assert [kwargs["enable_pdl"] for _, kwargs in atomic.calls] == [pdl] * 4


def _without_forward_verdicts(monkeypatch) -> None:
    """Clear the forward's shipped verdicts (the cluster SIMT windows of :data:`ROUTER_RULES` and of the tables of
    :data:`ROUTER_RULES_BY_SM`): no router table then gives the forward a verdict, so the forward without a split
    count takes the deterministic program's binding."""
    for table in (
        cake_backend.ROUTER_RULES,
        *cake_backend.ROUTER_RULES_BY_SM.values(),
    ):
        for key, rule in list(table.items()):
            if key[1] == "kk" and rule is not None:
                monkeypatch.setitem(table, key, None)


def _register(monkeypatch, arch: str, *templates: str) -> None:
    """Pretend this checkout registers ``templates`` on ``arch`` (CPU planner tests: nothing is loaded)."""
    table = dict(cake_jit.KERNELS.get(arch, {}))
    for template in templates:
        table[template] = "cake_dense_projection_gemm_" + "0" * 20
    monkeypatch.setitem(cake_jit.KERNELS, arch, table)


def _register_k1(monkeypatch, arch: str) -> None:
    _register(
        monkeypatch,
        arch,
        *(
            cake_backend.atomic_k1_template(bn, f32)
            for bn in cake_backend.K1_BLOCK_N_CHOICES
            for f32 in (False, True)
        ),
    )


# ---------------------------------------------------------------------------
# K1 planner (CPU)
# ---------------------------------------------------------------------------

# The weight-gradient rows of the deterministic package's benchmark and of the #7043 follow-ups: (d_in, d_out) of
# ``dW[d_out, d_in] = G[T, d_out].T @ X[T, d_in]``, each in BF16 and FP32 at two token counts (120 rows; the four MLA
# head rows, batched, make 124).
# fmt: off
WGRAD_SHAPES = (
    (2048, 4096), (2048, 6144), (2048, 16384), (4096, 32), (4096, 48), (4096, 64), (4096, 128), (4096, 512),
    (4096, 1024), (4096, 4096), (4096, 12288), (5120, 32), (5120, 48), (5120, 64), (5120, 128), (5120, 512),
    (5120, 1024), (5120, 5120), (5120, 17408), (6144, 32), (6144, 128), (6144, 576), (6144, 2048), (6144, 2048),
    (6144, 12288), (8192, 128), (12288, 4096), (12288, 6144), (16384, 6144), (17408, 5120),
)
# fmt: on
SM_COUNTS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 216}


@pytest.fixture
def no_k1_chains(monkeypatch):
    """Take :data:`K1_PARITY_RANGES_BY_SM` (``test_k1_parity_ranges_plan_their_program`` holds it) and
    :data:`K1_CHAIN_RANGES_BY_SM` out of the planner: their ranges come before every other K1 table
    (:func:`k1_chain_plan`), so a test of those tables holds them without it (``test_k1_chain_ranges`` holds the
    chain ranges).  The one-split ranges of :data:`K1_ONE_SPLIT_RANGES_BY_SM`, which come next, go too
    (``test_k1_152_sm_one_split_ranges`` holds them), and so do the cluster ranges of :data:`K1_DSM_RANGES_BY_SM`,
    which :func:`plan_atomic_gemm` reads before all of them (``test_k1_dsm_ranges`` holds them), and the fixed-order
    ranges of :data:`K1_ORDERED_RANGES_BY_SM`, which it reads next (``test_k1_ordered_ranges_plan_their_program``
    holds them)."""
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_ONE_SPLIT_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_ORDERED_RANGES_BY_SM", {})


@pytest.mark.parametrize(
    "arch, sm_count, rows",
    [
        ("sm_100a", 148, 28),
        ("sm_103a", 152, 26),
        ("sm_103a", 148, 28),
        ("sm_107a", 216, 15),
        ("sm_107a", 212, 14),
        ("sm_107a", 208, 11),
        ("sm_107a", 210, 0),
    ],
)
def test_k1_planners_take_the_measured_rows(arch, sm_count, rows):
    """The verdicts of the three architectures (on sm_107a with the chain ranges of 216 SMs) take 28 / 26 / 15 of the
    124 weight-gradient rows (the MLA head rows, ``L = 64``, run the deterministic program), at 16172 and 16231 tokens:
    148-SM sm_100a ``4096 -> 1024`` BF16 is not among them (its two splits there come from
    :data:`K1_ORDERED_RANGES_BY_SM`, which :func:`plan_atomic_gemm` reads before :func:`k1_plan`), 152-SM sm_103a
    ``5120 -> 64`` and ``5120 -> 128`` FP32 (and ``5120 -> 48`` BF16 at 16231 tokens) run the deterministic program
    there (:data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`), and the sm_107a rows that ran three or four splits run two
    splits or the deterministic program;
    on 148 SMs sm_103a leaves ``4096 -> 48`` to the deterministic program (28 rows).  16172 tokens span an odd number of
    64-token K blocks (253), below the first token count of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` (65538), so the
    FP32 rows of that table run their two-split program there, except sm_103a ``4096 -> 48`` on 152 SMs (its range from
    14338 to 32767 tokens).  On sm_107a the token counts of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` run the deterministic program (seven FP32 rows and one BF16 row at 16231
    tokens on 216 SMs; on 212 and 216 SMs outside their ranges in :data:`K1_CHAIN_RANGES_BY_SM`: 16231 tokens for
    ``4096 -> 512``, ``4096 -> 1024`` and ``5120 -> 512`` on 212 SMs and for ``4096 -> 512`` and ``4096 -> 1024`` on 216
    SMs, where the other five and six (row, token count) pairs run three accumulation chains; on 208 SMs ``4096 -> 512``
    and ``4096 -> 1024`` FP32 at 16231 tokens, where the other six pairs of the four one-split FP32 rows run one split).
    The per-SM-count sm_107a tables and the chain ranges take 14 rows on 212 SMs, the tables 11 on 208 SMs; an SM count
    below 216 without a table takes none."""
    taken = 0
    for d_in, d_out in WGRAD_SHAPES:
        for out_f32 in (False, True):
            for T in (16231, 16172):
                choice = cake_backend.k1_plan(
                    arch=arch,
                    L=1,
                    M=d_in,
                    N=d_out,
                    K=T,
                    out_f32=out_f32,
                    sm_count=sm_count,
                )
                if choice is not None:
                    block_n, splits = choice
                    assert block_n in cake_backend.K1_BLOCK_N_CHOICES and splits >= 1
                    taken += 1
    for d_in, d_out in ((192, 512), (512, 256)):
        assert (
            cake_backend.k1_plan(
                arch=arch,
                L=64,
                M=d_in,
                N=d_out,
                K=16231,
                out_f32=False,
                sm_count=sm_count,
            )
            is None
        )
    assert taken == rows


def test_k1_planner_choices():
    kw = dict(L=1, K=16231, out_f32=False)
    # sm_103a: 64-wide tiles, 128-wide for d_out = 128 from d_in 5120 up; a row without a verdict runs det
    assert cake_backend.k1_plan(arch="sm_103a", M=4096, N=32, sm_count=148, **kw) == (
        64,
        2,
    )
    # d_out 48 at d_in 4096: from 152 SMs only (K1_RULE_MIN_SM_COUNT)
    for out_f32 in (False, True):
        k48 = dict(arch="sm_103a", L=1, M=4096, N=48, K=16231, out_f32=out_f32)
        assert cake_backend.k1_plan(sm_count=152, **k48) == (64, 2)
        assert cake_backend.k1_plan(sm_count=148, **k48) is None
        # d_out 64 at d_in 4096 was measured and missed
        key = ("sm_103a", 1, 4096, 64, "f32" if out_f32 else "bf16")
        assert key in cake_backend.K1_SPLIT_RULES
        assert cake_backend.K1_SPLIT_RULES[key] is None
        assert (
            cake_backend.k1_plan(
                arch="sm_103a",
                M=4096,
                N=64,
                sm_count=152,
                L=1,
                K=16231,
                out_f32=out_f32,
            )
            is None
        )
    assert cake_backend.k1_plan(arch="sm_103a", M=6144, N=128, sm_count=152, **kw) == (
        128,
        2,
    )
    assert (
        cake_backend.k1_plan(arch="sm_103a", M=8192, N=128, sm_count=152, **kw) is None
    )
    # a thin row that was never measured runs the deterministic program
    assert (
        cake_backend.k1_plan(arch="sm_103a", M=2048, N=32, sm_count=152, **kw) is None
    )
    # sm_100a: (64, 2), on 148 SMs as on 152
    assert cake_backend.k1_plan(arch="sm_100a", M=5120, N=64, sm_count=148, **kw) == (
        64,
        2,
    )
    assert (
        cake_backend.k1_plan(arch="sm_100a", M=5120, N=128, sm_count=148, **kw) is None
    )
    # below the SM-count floor every row runs the deterministic program
    assert (
        cake_backend.k1_plan(arch="sm_100a", M=4096, N=64, sm_count=120, **kw) is None
    )
    # d_out = 128 at d_in 4096 was measured and missed on every architecture (152 SMs: FP32 two-split ranges of
    # K1_TWO_SPLIT_RANGES_BY_SM below the window, test_k1_152_sm_two_split_ranges)
    for arch, sms in SM_COUNTS.items():
        assert (
            cake_backend.k1_plan(arch=arch, M=4096, N=128, sm_count=sms, **kw) is None
        )
    # sm_107a: the 216-SM verdicts from 216 SMs; below, the table of the part's own SM count (208, 212) or none.
    # 4096 -> 32 BF16 ran four splits added in the order they finish on 216 SMs; it runs the deterministic program
    for sms in (208, 212, 216):
        assert (
            cake_backend.k1_plan(arch="sm_107a", M=4096, N=32, sm_count=sms, **kw)
            is None
        )
    assert (
        cake_backend.k1_plan(arch="sm_107a", M=4096, N=32, sm_count=210, **kw) is None
    )

    def p107(M, N, sms, out_f32=False):
        return cake_backend.k1_plan(
            arch="sm_107a", L=1, M=M, N=N, K=16231, out_f32=out_f32, sm_count=sms
        )

    for sms in (208, 212):
        # two splits (repeatable) where three ran before
        assert p107(4096, 32, sms, out_f32=True) == (64, 2)
        # 4096 -> 512 BF16 was measured and missed on every SM count
        assert p107(4096, 512, sms) is None
        # 4096 -> 32, 4096 -> 64, 5120 -> 32 BF16 and 6144 -> 32: no verdict on 208 and 212 SMs (their
        # deterministic program did not complete); on 216 SMs the rows that ran three or four splits run the
        # deterministic program
        for M, N in ((4096, 32), (4096, 64), (5120, 32), (6144, 32)):
            assert p107(M, N, sms) is None
    # 4096 -> 4096 FP32: the deterministic program on 208, 212 and 216 SMs
    assert p107(4096, 4096, 212, out_f32=True) is None
    assert p107(4096, 4096, 208, out_f32=True) is None
    assert p107(4096, 4096, 216, out_f32=True) is None
    # 6144 -> 32 FP32: the deterministic program on 208, 212 and 216 SMs
    assert p107(6144, 32, 212, out_f32=True) is None
    assert p107(6144, 32, 216, out_f32=True) is None
    assert p107(6144, 32, 208, out_f32=True) is None
    assert p107(8192, 128, 212) == (128, 2)
    assert p107(8192, 128, 208) is None
    # 5120 -> 1024 BF16: one split of 256-wide tiles on 216 SMs, except at 16231 tokens (the deterministic program);
    # on 208 it misses, on 212 it runs the deterministic program as well
    assert p107(5120, 1024, 212) is None
    assert p107(5120, 1024, 216) is None
    assert cake_backend.k1_plan(
        arch="sm_107a", M=5120, N=1024, sm_count=216, L=1, K=16172, out_f32=False
    ) == (256, 1)
    assert p107(5120, 1024, 208) is None
    assert (
        cake_backend.k1_plan(arch="sm_107a", M=4096, N=48, sm_count=216, **kw) is None
    )
    # 4096 -> 512 FP32: one split up to 40960 tokens, of three chains inside the ranges of K1_CHAIN_RANGES_BY_SM and of
    # one chain outside them; the deterministic program at 16231 and 16384 tokens and from 40961 tokens up (its entries
    # in K1_SPLIT_RULE_EXCLUDED_TOKENS) outside the chain ranges, which come first (the last ends at 40961)
    w512 = dict(arch="sm_107a", M=4096, N=512, sm_count=216, L=1, out_f32=True)
    for T in (960, 16173, 16300, 32767):
        assert cake_backend.k1_plan(K=T, **w512) == (128, 1), T
        assert cake_backend.k1_chain_plan(K=T, **w512) is None, T
    for T in (961, 16172, 32768, 40961):
        assert cake_backend.k1_plan(K=T, **w512) == (128, 1), T
        assert cake_backend.k1_chain_plan(K=T, **w512) == (128, 3), T
    for T in (16231, 16384, 40962, 45056, 65536):
        assert cake_backend.k1_plan(K=T, **w512) is None, T
        assert cake_backend.k1_chain_plan(K=T, **w512) is None, T
    # measured and missed on sm_107a: 4096 -> 64 BF16, 4096 -> 512 BF16, 4096 -> 4096 FP32
    for N, out_f32 in ((64, False), (512, False), (4096, True)):
        key = ("sm_107a", 1, 4096, N, "f32" if out_f32 else "bf16")
        assert key in cake_backend.K1_SPLIT_RULES
        assert cake_backend.K1_SPLIT_RULES[key] is None
        assert (
            cake_backend.k1_plan(
                arch="sm_107a",
                M=4096,
                N=N,
                sm_count=216,
                L=1,
                K=16231,
                out_f32=out_f32,
            )
            is None
        )
    for out_f32 in (False, True):
        key = ("sm_107a", 1, 4096, 128, "f32" if out_f32 else "bf16")
        assert cake_backend.K1_SPLIT_RULES[key] is None  # measured, missed
        assert (
            cake_backend.k1_plan(
                arch="sm_107a",
                M=4096,
                N=128,
                sm_count=216,
                L=1,
                K=16231,
                out_f32=out_f32,
            )
            is None
        )
    assert cake_backend.k1_plan(arch="sm_90a", M=4096, N=32, sm_count=132, **kw) is None


def test_k1_verdicts_name_registered_programs_and_fit_one_wave():
    """Every K1 verdict names a registered template of its architecture and a split count its token counts allow;
    the sm_100a / sm_103a verdicts fit one wave of the 74 CTA pairs of the SM-count floor (148 SMs)."""
    for (arch, L, M, N, dt), choice in cake_backend.K1_SPLIT_RULES.items():
        assert arch in cake_backend.K1_MIN_SM_COUNT and dt in ("bf16", "f32") and L == 1
        if choice is None:
            continue
        block_n, splits = choice
        assert block_n in cake_backend.K1_BLOCK_N_CHOICES and splits >= 1
        template = cake_backend.atomic_k1_template(block_n, dt == "f32")
        assert template in cake_jit.KERNELS.get(arch, {}), (arch, template)
        if arch in ("sm_100a", "sm_103a"):
            assert splits == 2
            tiles = -(-M // cake_backend.K1_PAIR_ROWS) * -(-N // block_n)
            assert tiles * splits <= cake_backend.K1_MIN_SM_COUNT[arch] // 2
    # a row's own SM-count floor names a row with a verdict and lies above its architecture's floor
    for key, floor in cake_backend.K1_RULE_MIN_SM_COUNT.items():
        assert cake_backend.K1_SPLIT_RULES.get(key) is not None, key
        assert floor > cake_backend.K1_MIN_SM_COUNT[key[0]], key
    # a per-SM-count table acts only below the SM-count floor of each of its rows (it never replaces a verdict of
    # K1_SPLIT_RULES on the parts that table was measured on) and names registered templates
    for (arch, sm_count), table in cake_backend.K1_SPLIT_RULES_BY_SM.items():
        for key, choice in table.items():
            key_arch, L, M, N, dt = key
            assert key_arch == arch and dt in ("bf16", "f32") and L == 1
            floor = max(
                cake_backend.K1_MIN_SM_COUNT[arch],
                cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
            )
            assert sm_count < floor, (arch, sm_count, key)
            if choice is None:
                continue
            block_n, splits = choice
            assert block_n in cake_backend.K1_BLOCK_N_CHOICES and splits >= 1
            template = cake_backend.atomic_k1_template(block_n, dt == "f32")
            assert template in cake_jit.KERNELS.get(arch, {}), (
                arch,
                sm_count,
                template,
            )
    # an override acts from the SM-count floor of each of its rows up (below it the per-SM-count tables decide),
    # replaces a row of K1_SPLIT_RULES and names a registered template
    for (arch, sm_count), table in cake_backend.K1_SPLIT_OVERRIDES_BY_SM.items():
        for key, choice in table.items():
            key_arch, L, M, N, dt = key
            assert key_arch == arch and dt in ("bf16", "f32") and L == 1
            assert key in cake_backend.K1_SPLIT_RULES, key
            floor = max(
                cake_backend.K1_MIN_SM_COUNT[arch],
                cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
            )
            assert sm_count >= floor, (arch, sm_count, key)
            if choice is None:
                continue
            block_n, splits = choice
            assert block_n in cake_backend.K1_BLOCK_N_CHOICES and splits >= 1
            template = cake_backend.atomic_k1_template(block_n, dt == "f32")
            assert template in cake_jit.KERNELS.get(arch, {}), (
                arch,
                sm_count,
                template,
            )


def test_k1_small_token_counts_run_the_deterministic_program(monkeypatch):
    """``num_splits * 64 <= T + 63``: a token count too short for the split count runs the deterministic program."""
    kw = dict(arch="sm_107a", L=1, M=4096, N=32, out_f32=False, sm_count=216)
    monkeypatch.setitem(
        cake_backend.K1_SPLIT_RULES, ("sm_107a", 1, 4096, 32, "bf16"), (64, 4)
    )
    assert cake_backend.k1_plan(K=193, **kw) == (64, 4)
    assert cake_backend.k1_plan(K=192, **kw) is None
    kw = dict(arch="sm_103a", L=1, M=4096, N=32, out_f32=False, sm_count=152)
    assert cake_backend.k1_plan(K=65, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=64, **kw) is None
    assert cake_backend.k1_splits_fit(1, 1) and not cake_backend.k1_splits_fit(0, 64)


def test_k1_excluded_token_counts(no_k1_chains):
    """A row of :data:`K1_SPLIT_RULES` or of a table of :data:`K1_SPLIT_RULES_BY_SM` runs the deterministic program at
    the token counts of its :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` entries: from the entry's SM count up where that count
    is at or above the row's SM-count floor, on exactly that SM count below it.  Every entry belongs to a row with a
    verdict on its part; per SM count its ranges are closed, disjoint and not adjacent, start at a measured token count
    or one token above one (512 to 65537; the sm_107a FP32 one-split rows from 40961, one token above the last measured
    token count at which they meet the accuracy budget, ``5120 -> 1024`` from 32769) and end at one or have no upper
    bound; the row runs the deterministic program at both ends and keeps its verdict (inside the token window, the
    override of :data:`K1_SPLIT_OVERRIDES_BY_SM` for the part's SM count, if any) one token outside a range (inside the
    token window of a per-SM-count table), on a part with fewer SMs than an entry at or above the floor, on the floor's
    part for an entry below it, and above 65536 tokens unless a range has no upper bound.  A token count with an odd
    number of 64-token blocks inside an entry of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` for the part, or inside an
    entry of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` for the part's exact SM count, runs the deterministic program
    too."""
    table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    odd_table = cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS
    exact_table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM
    by_sm = cake_backend.K1_SPLIT_RULES_BY_SM
    overrides = cake_backend.K1_SPLIT_OVERRIDES_BY_SM
    routed = {k for k, v in cake_backend.K1_SPLIT_RULES.items() if v is not None}
    for rows in by_sm.values():
        routed |= {k for k, v in rows.items() if v is not None}
    assert table and set(table) <= routed
    lo_t, hi_t = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    for key, entries in table.items():
        arch, L, M, N, dt = key
        floor = max(
            cake_backend.K1_MIN_SM_COUNT[arch],
            cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
        )
        kw = dict(arch=arch, L=L, M=M, N=N, out_f32=dt == "f32")
        assert entries, key
        spans_of = {}
        for sms, a, b in entries:
            spans_of.setdefault(sms, []).append((a, b))
        for spans in spans_of.values():
            spans.sort()
            for (_lo, b), (c, _hi) in zip(spans, spans[1:], strict=False):
                assert b is not None and b + 1 < c, key

        def cut(sm_count, T, key=key, floor=floor):
            odd = odd_table.get(key, ()) if -(-T // 64) % 2 else ()
            for sms, a, b in table[key] + odd:
                on_part = sm_count >= sms if sms >= floor else sm_count == sms
                if on_part and a <= T and (b is None or b >= T):
                    return True
            exact = exact_table.get((key[0], sm_count), {}).get(key, ())
            return any(a <= T and (b is None or b >= T) for a, b in exact)

        def kept(sm_count, T, want, key=key, floor=floor):
            override = overrides.get((key[0], sm_count), {}).get(key)
            if sm_count >= floor and override is not None and lo_t <= T <= hi_t:
                return tuple(override)
            return want

        for sms, a, b in entries:
            assert 512 <= a <= 65537, key
            assert b is None or a <= b <= 65536, key
            if sms >= floor:
                verdict = cake_backend.K1_SPLIT_RULES[key]
                parts = (sms, sms + 64)
            else:
                verdict = by_sm[(arch, sms)][key]
                parts = (sms,)
            assert verdict is not None, (key, sms)
            want = tuple(verdict)
            ends = (a,) if b is None else (a, b)
            for sm_count in parts:
                for T in ends + ((65537, 131072) if b is None else ()):
                    got = cake_backend.k1_plan(K=T, sm_count=sm_count, **kw)
                    assert got is None, (key, sm_count, T)
                for T in (a - 1,) + (() if b is None else (b + 1,)):
                    if cut(sm_count, T) or (sms < floor and not lo_t <= T <= hi_t):
                        continue
                    got = cake_backend.k1_plan(K=T, sm_count=sm_count, **kw)
                    assert got == kept(sm_count, T, want), (key, sm_count, T)
            if sms >= floor and sms - 1 >= floor:
                for T in ends:
                    got = cake_backend.k1_plan(K=T, sm_count=sms - 1, **kw)
                    want_t = None if cut(sms - 1, T) else kept(sms - 1, T, want)
                    assert got == want_t, (key, T)
            if sms < floor and cake_backend.K1_SPLIT_RULES.get(key) is not None:
                for T in ends:
                    if not cut(floor, T):
                        got = cake_backend.k1_plan(K=T, sm_count=floor, **kw)
                        rule = tuple(cake_backend.K1_SPLIT_RULES[key])
                        assert got == kept(floor, T, rule), (key, T)
        top = max(floor, *(e[0] for e in entries))
        verdict = cake_backend.K1_SPLIT_RULES.get(key)
        if verdict is not None and not any(e[2] is None for e in entries):
            for T in (65537, 65601):
                got = cake_backend.k1_plan(K=T, sm_count=top, **kw)
                assert got == (None if cut(top, T) else tuple(verdict)), (key, T)
    kw = dict(arch="sm_100a", L=1, M=4096, N=32, out_f32=False)
    assert cake_backend.k1_plan(K=512, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=511, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=512, sm_count=151, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=65535, sm_count=152, **kw) == (64, 2)
    for sm_count in (148, 152):
        assert cake_backend.k1_plan(K=65536, sm_count=sm_count, **kw) is None
        assert cake_backend.k1_plan(K=131072, sm_count=sm_count, **kw) is None
    kw = dict(arch="sm_103a", L=1, M=4096, N=32, out_f32=False)
    assert cake_backend.k1_plan(K=512, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=511, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=512, sm_count=151, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=65535, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=131072, sm_count=152, **kw) is None
    kw = dict(arch="sm_103a", L=1, M=5120, N=32, out_f32=False)
    assert cake_backend.k1_plan(K=512, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=511, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=512, sm_count=151, **kw) == (64, 2)
    for T in (32768, 45057):
        assert cake_backend.k1_plan(K=T, sm_count=152, **kw) == (64, 2), T
    assert cake_backend.k1_plan(K=45058, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=131072, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=131072, sm_count=149, **kw) == (64, 2)
    # a row without an entry keeps its verdict at every token count its split count fits
    key = ("sm_100a", 1, 6144, 32, "f32")
    kw = dict(arch="sm_100a", L=1, M=6144, N=32, out_f32=True, sm_count=152)
    assert (
        key not in table and key not in cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS
    )
    for T in (65, 512, 16173, 16231, 65536, 65537, 131072):
        assert cake_backend.k1_plan(K=T, **kw) == (64, 2), T


def test_k1_rule_table_entries_decide_the_row(monkeypatch):
    # 16384 tokens (an even number of 64-token K blocks): 16231 is a 216-SM token count of K1_SPLIT_RULE_EXCLUDED_TOKENS
    # for sm_107a 4096 -> 32 FP32
    kw = dict(L=1, M=4096, N=32, K=16384, out_f32=True)
    key = ("sm_103a", 1, 4096, 32, "f32")
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=152, **kw) == (64, 2)
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, None)
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=152, **kw) is None
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (128, 3))
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=152, **kw) == (128, 3)
    # the T guard applies to a verdict too
    assert (
        cake_backend.k1_plan(
            arch="sm_103a", sm_count=152, L=1, M=4096, N=32, K=128, out_f32=True
        )
        is None
    )
    # a row's own minimum SM count applies on top of the architecture's
    monkeypatch.setitem(cake_backend.K1_RULE_MIN_SM_COUNT, key, 160)
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=152, **kw) is None
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=160, **kw) == (128, 3)
    # a verdict of an architecture with a minimum SM count applies from that count only
    key107 = ("sm_107a", 1, 4096, 32, "f32")
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key107, (128, 2))
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=216, **kw) == (128, 2)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=210, **kw) is None
    # below that count the part's own table decides (K1_SPLIT_RULES_BY_SM), not the 216-SM verdict
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=212, **kw) == (64, 2)
    monkeypatch.setitem(cake_backend.K1_MIN_SM_COUNT, "sm_107a", 0)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=212, **kw) == (128, 2)


def test_k1_per_sm_count_tables_decide_only_their_own_sm_count(monkeypatch):
    """Below the architecture's SM-count floor a part takes the table of :data:`K1_SPLIT_RULES_BY_SM` for its exact
    SM count; another SM count below the floor runs the deterministic program, and from the floor up the verdicts of
    :data:`K1_SPLIT_RULES` decide as before (a per-SM-count table never changes the 216-SM routing).  At 16172 tokens:
    16231 is a 216-SM token count of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for ``4096 -> 32`` FP32."""
    kw = dict(L=1, M=4096, N=32, K=16172, out_f32=True)
    key = ("sm_107a", 1, 4096, 32, "f32")
    tables = dict(cake_backend.K1_SPLIT_RULES_BY_SM)
    tables[("sm_107a", 210)] = {key: (64, 2)}
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RULES_BY_SM", tables)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=210, **kw) == (64, 2)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=211, **kw) is None
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=209, **kw) is None
    assert (
        cake_backend.k1_plan(arch="sm_107a", sm_count=216, **kw)
        == cake_backend.K1_SPLIT_RULES[key]
    )
    # a None entry of the table keeps the row on the deterministic program; the T guard applies to its verdicts too
    tables[("sm_107a", 210)] = {key: None}
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=210, **kw) is None
    tables[("sm_107a", 210)] = {key: (64, 4)}
    short = dict(kw, K=192)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=210, **short) is None
    # a row below its own K1_RULE_MIN_SM_COUNT (above the architecture's floor) also reads the part's table
    k48 = ("sm_103a", 1, 4096, 48, "f32")
    kw48 = dict(kw, N=48)
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=148, **kw48) is None
    tables[("sm_103a", 148)] = {k48: (64, 2)}
    assert cake_backend.k1_plan(arch="sm_103a", sm_count=148, **kw48) == (64, 2)


@pytest.mark.parametrize("sm_count", [208, 212])
def test_k1_per_sm_count_tables_hold_only_at_their_token_counts(sm_count, no_k1_chains):
    """The tables of :data:`K1_SPLIT_RULES_BY_SM` were measured at 16172 to 16384 tokens and hold there
    (:data:`K1_SPLIT_RULES_BY_SM_TOKENS`): every row of the 208-SM and 212-SM tables takes its verdict at the first
    and the last token count of that window (except a token count of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the
    part) and runs the deterministic program one token outside it, unless a range of the row in
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` holds that token count (the next test)."""
    assert cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS == (16172, 16384)
    table = cake_backend.K1_SPLIT_RULES_BY_SM[("sm_107a", sm_count)]
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    ranges = cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES.get(
        ("sm_107a", sm_count), {}
    )
    routed = 0
    for key, choice in sorted(table.items()):
        arch, L, M, N, dt = key
        kw = dict(arch=arch, L=L, M=M, N=N, out_f32=dt == "f32", sm_count=sm_count)
        want = None if choice is None else tuple(choice)
        cut = {
            T
            for sms, lo, hi in excluded.get(key, ())
            for T in (16172, 16384)
            if sms == sm_count and lo <= T and (hi is None or hi >= T)
        }
        for T in (16172, 16384):
            got = cake_backend.k1_plan(K=T, **kw)
            assert got == (None if T in cut else want), (M, N, dt, T)
        for T in (16171, 16385):
            held = any(lo <= T <= hi for lo, hi in ranges.get(key, ()))
            got = cake_backend.k1_plan(K=T, **kw)
            assert got == (want if held else None), (M, N, dt, T)
        routed += want is not None
    assert routed > 0
    if sm_count == 212:
        # 4096 -> 32 BF16 has no verdict on 212 SMs (its deterministic program did not complete next to a resident
        # kernel in long-lived processes): the deterministic program at every token count
        kw = dict(arch="sm_107a", L=1, M=4096, N=32, out_f32=False, sm_count=212)
        assert table[("sm_107a", 1, 4096, 32, "bf16")] is None
        for T in (16171, 16172, 16231, 16384, 16385):
            assert cake_backend.k1_plan(K=T, **kw) is None, T


#: The token ranges of :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` on 212-SM sm_107a parts, ``(d_in, d_out, out_dtype)
#: -> ranges``: the measured token counts (512, 1024, 2048, 4096, 8192, 32768, 65536 and the window's 16172 and 16384)
#: at which the row's program passed on both nodes, joined up to a failing or unmeasured one.
K1_212_TOKEN_RANGES = {
    (4096, 512, "f32"): ((512, 512),),
    (4096, 1024, "f32"): ((512, 512),),
    (5120, 512, "f32"): ((512, 512),),
    (5120, 1024, "f32"): ((512, 512),),
    (8192, 128, "bf16"): ((1024, 16172), (16384, 32768)),
    (8192, 128, "f32"): ((1024, 16172), (16384, 32768)),
}
#: The measured token counts of those rows: 512 to 65536 tokens and the window of
#: :data:`K1_SPLIT_RULES_BY_SM_TOKENS`.
K1_212_MEASURED_TOKENS = (512, 1024, 2048, 4096, 8192, 16172, 16384, 32768, 65536)


def test_k1_212_sm_token_ranges(no_k1_chains):
    """On 212-SM sm_107a parts the rows of :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` take their 212-SM verdict at
    every token count of their ranges (both ends and the measured token counts inside), except a token count of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part, and run the deterministic program one token outside a range
    and at every measured token count outside the ranges and the window (except the two-split ranges of
    :data:`K1_TWO_SPLIT_RANGES_BY_SM`: ``test_k1_212_sm_two_split_ranges``).  Every range starts and ends at a
    measured token count, lies inside 512 to 65536 tokens and belongs to a row with a program on that part.  Another
    SM count (208, 210, 216, 217) and every other row keep their routing: 208 SMs and the other 212-SM rows run the
    deterministic program outside the window, 216 SMs and up decide by :data:`K1_SPLIT_RULES`."""
    table = cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES
    assert set(table) == {("sm_107a", 212)}
    by_sm = cake_backend.K1_SPLIT_RULES_BY_SM[("sm_107a", 212)]
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    rows = table[("sm_107a", 212)]
    assert {(k[2], k[3], k[4]) for k in rows} == set(K1_212_TOKEN_RANGES)

    def cut(key, sm_count, T):
        floor = cake_backend.K1_MIN_SM_COUNT["sm_107a"]
        return any(
            (sm_count >= sms if sms >= floor else sm_count == sms)
            and lo <= T
            and (hi is None or hi >= T)
            for sms, lo, hi in excluded.get(key, ())
        )

    def p(key, sm_count, T):
        arch, L, M, N, dt = key
        return cake_backend.k1_plan(
            arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sm_count
        )

    for key, spans in rows.items():
        assert key[0] == "sm_107a" and key[1] == 1
        assert tuple(spans) == K1_212_TOKEN_RANGES[(key[2], key[3], key[4])], key
        verdict = by_sm[key]
        assert verdict is not None, key
        want = tuple(verdict)
        for (_, b), (c, _) in zip(spans, spans[1:], strict=False):
            assert b < c, key
        held = set()
        for lo, hi in spans:
            assert lo in K1_212_MEASURED_TOKENS and hi in K1_212_MEASURED_TOKENS, key
            assert 512 <= lo <= hi <= 65536, key
            inside = {lo, hi} | {T for T in K1_212_MEASURED_TOKENS if lo <= T <= hi}
            held |= inside
            for T in sorted(inside):
                got = p(key, 212, T)
                assert got == (None if cut(key, 212, T) else want), (key, T)
            for T in (lo - 1, hi + 1):
                if first <= T <= last or any(a <= T <= b for a, b in spans):
                    continue
                assert p(key, 212, T) is None, (key, T)
        two = K1_212_TWO_SPLIT_RANGES.get((key[2], key[3], key[4]), ())
        for T in K1_212_MEASURED_TOKENS:
            two_split = any(a <= T <= b for a, b, _ in two)
            if T not in held and not first <= T <= last and not two_split:
                assert p(key, 212, T) is None, (key, T)
        for T in (131072, 511):
            assert p(key, 212, T) is None, (key, T)
        # the other parts keep their routing at every one of these token counts
        rule = cake_backend.K1_SPLIT_RULES.get(key)
        for T in sorted(held):
            if not first <= T <= last:
                assert p(key, 208, T) is None, (key, T)
                assert p(key, 210, T) is None, (key, T)
            for sm_count in (216, 217):
                got = p(key, sm_count, T)
                over = cake_backend.K1_SPLIT_OVERRIDES_BY_SM.get(
                    ("sm_107a", sm_count), {}
                )
                rule_t = over.get(key, rule) if first <= T <= last else rule
                expect = (
                    None if rule_t is None or cut(key, sm_count, T) else tuple(rule_t)
                )
                assert got == expect, (key, sm_count, T)
    # a 212-SM row without ranges runs the deterministic program outside the window (the rows of
    # K1_SPLIT_RANGE_OVERRIDES_BY_SM: the next test)
    overrides = cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM.get(("sm_107a", 212), {})
    for key, verdict in by_sm.items():
        if key in rows or key in overrides or verdict is None:
            continue
        for T in (512, 1024, 8192, 32768, 65536):
            assert p(key, 212, T) is None, (key, T)


#: The ranges of :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` on 212-SM sm_107a parts, ``(d_in, d_out, out_dtype) ->
#: ((min_tokens, max_tokens, (block_n, num_splits)), ...)``: measured as :data:`K1_212_TOKEN_RANGES` (512, 1024, 2048,
#: 4096, 8192, 12288 and 32768 tokens for ``5120 -> 1024`` BF16).
K1_212_RANGE_OVERRIDES = {
    (5120, 1024, "bf16"): ((512, 12288, (256, 1)),),
}
K1_212_OVERRIDE_MEASURED_TOKENS = (512, 1024, 2048, 4096, 8192, 12288, 32768)


def test_k1_212_sm_range_overrides():
    """On 212-SM sm_107a parts a token count of a range of :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` runs that range's
    program (both ends and the measured token counts inside), except a token count of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part; one token outside it the row takes what the window, its ranges
    in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and its 212-SM verdict give (the deterministic program for these
    rows).  A range lies inside 512 to 65536 tokens, outside the window and outside the row's token ranges, and the
    window keeps the row's 212-SM verdict (``5120 -> 1024`` BF16: the deterministic program).  208, 210, 216 and 217
    SMs keep their routing at every one of these token counts."""
    table = cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM
    assert set(table) == {("sm_107a", 212)}
    rows = table[("sm_107a", 212)]
    assert {(k[2], k[3], k[4]) for k in rows} == set(K1_212_RANGE_OVERRIDES)
    by_sm = cake_backend.K1_SPLIT_RULES_BY_SM[("sm_107a", 212)]
    token_ranges = cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES[("sm_107a", 212)]
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS

    def cut(key, sm_count, T):
        floor = cake_backend.K1_MIN_SM_COUNT["sm_107a"]
        return any(
            (sm_count >= sms if sms >= floor else sm_count == sms)
            and lo <= T
            and (hi is None or hi >= T)
            for sms, lo, hi in excluded.get(key, ())
        )

    def p(key, sm_count, T):
        arch, L, M, N, dt = key
        return cake_backend.k1_plan(
            arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sm_count
        )

    for key, spans in rows.items():
        assert key[0] == "sm_107a" and key[1] == 1
        assert tuple(spans) == K1_212_RANGE_OVERRIDES[(key[2], key[3], key[4])], key
        own = token_ranges.get(key, ())
        verdict = by_sm.get(key)
        table_plan = None if verdict is None else tuple(verdict)
        for lo, hi, choice in spans:
            assert 512 <= lo <= hi <= 65536, key
            assert lo in K1_212_OVERRIDE_MEASURED_TOKENS, key
            assert hi in K1_212_OVERRIDE_MEASURED_TOKENS, key
            assert hi < first or lo > last, key
            assert not any(a <= hi and lo <= b for a, b in own), key
            assert tuple(choice) != table_plan, key
            inside = {lo, hi} | {
                T for T in K1_212_OVERRIDE_MEASURED_TOKENS if lo <= T <= hi
            }
            for T in sorted(inside):
                want = None if cut(key, 212, T) else tuple(choice)
                assert p(key, 212, T) == want, (key, T)
                assert p(key, 208, T) is None, (key, T)
                assert p(key, 210, T) is None, (key, T)
                rule = cake_backend.K1_SPLIT_RULES.get(key)
                for sm_count in (216, 217):
                    expect = (
                        None if rule is None or cut(key, sm_count, T) else tuple(rule)
                    )
                    assert p(key, sm_count, T) == expect, (key, sm_count, T)
            for T in (lo - 1, hi + 1):
                if any(a <= T <= b for a, b, _ in spans):
                    continue
                held = first <= T <= last or any(a <= T <= b for a, b in own)
                want = table_plan if held and not cut(key, 212, T) else None
                assert p(key, 212, T) == want, (key, T)
        for T in (first, 16231, last):
            want = None if table_plan is None or cut(key, 212, T) else table_plan
            assert p(key, 212, T) == want, (key, T)
        for T in (511, 131072):
            assert p(key, 212, T) is None, (key, T)
    # 5120 -> 1024 BF16: the deterministic program in the window and at the measured token counts above the range
    key = ("sm_107a", 1, 5120, 1024, "bf16")
    assert by_sm[key] is None
    for T in (12289, 16172, 16231, 16384, 32768, 65536):
        assert p(key, 212, T) is None, T


#: The ranges of :data:`K1_TWO_SPLIT_RANGES_BY_SM` on 212-SM sm_107a parts, ``(d_in, d_out, out_dtype) ->
#: ((min_tokens, max_tokens, (block_n, num_splits)), ...)``: two splits of the row's 212-SM tile width at every token
#: count of the range (either number of 64-token K blocks).
K1_212_TWO_SPLIT_RANGES = {
    (4096, 512, "f32"): ((1024, 8192, (128, 2)), (32768, 65536, (128, 2))),
    (4096, 1024, "f32"): ((2048, 8192, (192, 2)), (32768, 65536, (192, 2))),
    (5120, 512, "f32"): ((1024, 8192, (128, 2)), (32768, 65536, (128, 2))),
    (5120, 1024, "f32"): ((1024, 8192, (256, 2)),),
}
#: The measured token counts of those rows: 512 to 65536 tokens and the window's 16231 and 16384.
K1_212_TWO_SPLIT_MEASURED_TOKENS = (
    512,
    1024,
    2048,
    4096,
    8192,
    16231,
    16384,
    32768,
    65536,
)


def _k1_blocks(T):
    """The 64-token K blocks a token count spans."""
    return -(-int(T) // 64)


def test_k1_212_sm_two_split_ranges(monkeypatch, no_k1_chains):
    """On 212-SM sm_107a parts every token count of a range of :data:`K1_TWO_SPLIT_RANGES_BY_SM` (an even or an odd
    number of 64-token K blocks: 1025, 2049, 6000, 40961) runs that range's program (two splits of the row's 212-SM tile
    width), except a token count of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part; every token count outside the
    ranges keeps the plan the part has without the table.  A range starts and ends at a measured token count, lies
    inside 1024 to 65536 tokens, outside the window and outside the row's ranges in
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM`.  208, 210, 211, 213, 216 and
    217 SMs plan every row and token count as without the table."""
    table = cake_backend.K1_TWO_SPLIT_RANGES_BY_SM
    assert set(table) == {("sm_107a", 212)} | set(K1_152_TWO_SPLIT_RANGES)
    # 4096 -> 128: test_k1_212_sm_4096_to_128_two_split_range
    narrow = ("sm_107a", 1, 4096, 128, "f32")
    rows = {k: v for k, v in table[("sm_107a", 212)].items() if k != narrow}
    assert set(table[("sm_107a", 212)]) == set(rows) | {narrow}
    assert {(k[2], k[3], k[4]) for k in rows} == set(K1_212_TWO_SPLIT_RANGES)
    by_sm = cake_backend.K1_SPLIT_RULES_BY_SM[("sm_107a", 212)]
    token_ranges = cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES[("sm_107a", 212)]
    overrides = cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM.get(("sm_107a", 212), {})
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS

    def cut(key, sm_count, T):
        floor = cake_backend.K1_MIN_SM_COUNT["sm_107a"]
        return any(
            (sm_count >= sms if sms >= floor else sm_count == sms)
            and lo <= T
            and (hi is None or hi >= T)
            for sms, lo, hi in excluded.get(key, ())
        )

    def p(M, N, T, sm_count=212):
        return cake_backend.k1_plan(
            arch="sm_107a", L=1, M=M, N=N, K=T, out_f32=True, sm_count=sm_count
        )

    for key, spans in rows.items():
        assert key[0] == "sm_107a" and key[1] == 1 and key[4] == "f32", key
        want = K1_212_TWO_SPLIT_RANGES[(key[2], key[3], key[4])]
        assert tuple(spans) == want, key
        verdict = by_sm[key]
        assert verdict is not None, key
        for (_a, b, _v), (c, _d, _w) in zip(spans, spans[1:], strict=False):
            assert b < c, key
        for lo, hi, choice in spans:
            assert lo in K1_212_TWO_SPLIT_MEASURED_TOKENS, key
            assert hi in K1_212_TWO_SPLIT_MEASURED_TOKENS, key
            assert 1024 <= lo <= hi <= 65536, key
            assert hi < first or lo > last, key
            own = token_ranges.get(key, ())
            assert not any(a <= hi and lo <= b for a, b in own), key
            over = overrides.get(key, ())
            assert not any(a <= hi and lo <= b for a, b, _ in over), key
            assert tuple(choice) == (verdict[0], 2), key
    # inside a range the two-split program at an odd number of K blocks (17, 33, 641 blocks) as at an even number
    # (16, 94, 128, 512, 1024 blocks); one token outside a range the deterministic program
    for M, N in ((4096, 512), (5120, 512)):
        for T in (1024, 1025, 6000, 8192, 32768, 40961, 65536):
            assert p(M, N, T) == (128, 2), (M, N, T)
        for T in (8193, 32767, 65537):
            assert p(M, N, T) is None, (M, N, T)
    for T in (1024, 8193, 32767, 65537):
        assert p(4096, 1024, T) is None, T
    for T in (2048, 2049, 6000, 40961, 65536):
        assert p(4096, 1024, T) == (192, 2), T
    for T in (1023, 8193, 32768, 65536):
        assert p(5120, 1024, T) is None, T
    for T in (1024, 1025, 6000, 8192):
        assert p(5120, 1024, T) == (256, 2), T
    # outside the ranges the rows keep their 212-SM routing: the one-split program at 512 tokens
    # (K1_SPLIT_RULES_BY_SM_TOKEN_RANGES), the deterministic program elsewhere outside the window
    for M, N, _ in K1_212_TWO_SPLIT_RANGES:
        assert p(M, N, 512) == tuple(by_sm[("sm_107a", 1, M, N, "f32")]), (M, N)
        for T in (8320, 12000, 32640, 65664, 131072):
            assert p(M, N, T) is None, (M, N, T)
        # 208 SMs: the deterministic program outside the window; 216 SMs: the one-split verdict below the row's
        # open-ended entry (``K1_SM107A_WIDE_FP32_OPEN``: 40961 tokens, ``5120 -> 1024`` 32769), the deterministic
        # program from there up
        rule = tuple(cake_backend.K1_SPLIT_RULES[("sm_107a", 1, M, N, "f32")])
        T0 = K1_SM107A_WIDE_FP32_OPEN[(M, N)]
        for T in (960, 1024, 6000, 8192, 32768, T0 - 1):
            assert p(M, N, T, sm_count=216) == rule, (M, N, T)
        for T in (1024, 6000, 8192, 32768):
            assert p(M, N, T, sm_count=208) is None, (M, N, T)
        for T in (T0, 40961, 65536):
            assert p(M, N, T, sm_count=216) is None, (M, N, T)

    sms_list = (208, 210, 211, 212, 213, 216, 217)
    Ts = {
        T + d
        for T in K1_212_TWO_SPLIT_MEASURED_TOKENS + (6000, 12000, 40961, 131072)
        for d in (-1, 0, 1)
    }
    for spans in rows.values():
        for lo, hi, _ in spans:
            Ts |= {lo + 63, lo + 64, lo + 65, hi - 64, hi - 63, (lo + hi) // 2}

    def plans():
        out = {}
        for key in rows:
            _, L, M, N, dt = key
            for sm_count in sms_list:
                for T in sorted(Ts):
                    out[(key, sm_count, T)] = cake_backend.k1_plan(
                        arch="sm_107a",
                        L=L,
                        M=M,
                        N=N,
                        K=T,
                        out_f32=dt == "f32",
                        sm_count=sm_count,
                    )
        return out

    with_table = plans()
    monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
    without = plans()
    routed = 0
    for (key, sm_count, T), plan in with_table.items():
        held = [tuple(v) for lo, hi, v in rows[key] if lo <= T <= hi]
        two_split = sm_count == 212 and bool(held)
        if two_split and not cut(key, sm_count, T):
            assert plan == held[0], (key, T)
            assert without[(key, sm_count, T)] is None, (key, T)
            routed += 1
        else:
            assert plan == without[(key, sm_count, T)], (key, sm_count, T)
    assert routed


#: The ranges of :data:`K1_TWO_SPLIT_RANGES_BY_SM` on 152-SM sm_100a and sm_103a parts, ``(arch, sm_count) ->
#: ((min_tokens, max_tokens, (block_n, num_splits)), ...)`` of the FP32 row ``4096 -> 128``: two splits of 64-wide
#: tiles at every token count of the range (either number of 64-token K blocks).
K1_152_TWO_SPLIT_RANGES = {
    ("sm_100a", 152): ((1024, 4096, (64, 2)),),
    ("sm_103a", 152): ((1024, 6144, (64, 2)),),
}
#: The measured token counts of that row on 152 SMs.
K1_152_TWO_SPLIT_MEASURED_TOKENS = (512, 1024, 1088, 2048, 2112, 4096, 6144, 8192)


def test_k1_152_sm_two_split_ranges(monkeypatch, no_k1_chains):
    """On 152-SM sm_100a and sm_103a parts (at or above the SM-count floor) the FP32 row ``4096 -> 128``, which has no
    verdict in :data:`K1_SPLIT_RULES`, runs two splits of 64-wide tiles at every token count of its range in
    :data:`K1_TWO_SPLIT_RANGES_BY_SM`, at an even or an odd number of 64-token K blocks (1025, 1088, 2112, 4033): 1024
    to 4096 tokens on sm_100a, 1024 to 6144 on sm_103a.  A token count outside the range (512, 960, 1023, 4097 / 6145,
    8192), BF16, and every other SM count (148, 150, 151, 153, 160) run the deterministic program as without the table.
    A range starts and ends at a measured token count and lies outside the window of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS`."""
    table = cake_backend.K1_TWO_SPLIT_RANGES_BY_SM
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS

    def p(arch, T, sm_count=152, out_f32=True):
        return cake_backend.k1_plan(
            arch=arch, L=1, M=4096, N=128, K=T, out_f32=out_f32, sm_count=sm_count
        )

    for part, want in K1_152_TWO_SPLIT_RANGES.items():
        arch, sms = part
        key = (arch, 1, 4096, 128, "f32")
        assert table[part] == {key: want}, part
        assert cake_backend.K1_SPLIT_RULES[key] is None, key
        floor = max(
            cake_backend.K1_MIN_SM_COUNT[arch],
            cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
        )
        assert floor <= sms, key
        for lo, hi, choice in want:
            assert lo in K1_152_TWO_SPLIT_MEASURED_TOKENS, key
            assert hi in K1_152_TWO_SPLIT_MEASURED_TOKENS, key
            assert hi < first or lo > last, key
            assert tuple(choice) == (64, 2), key
    # the measured token counts: two splits at 1024, 1088, 2048, 2112 and 4096 tokens (and 6144 on sm_103a), the
    # deterministic program at 512 and 8192 tokens
    det_ts = (512, 960, 1023, 8192, 16231)
    for arch, top in (("sm_100a", 4096), ("sm_103a", 6144)):
        for T in (1024, 1025, 1088, 1100, 2048, 2112, 3072, 4033, 4096, top):
            assert p(arch, T) == (64, 2), (arch, T)
        for T in det_ts + (top + 1, top + 64, top + 65):
            assert p(arch, T) is None, (arch, T)
        for T in (1024, 2048, 4096, top):
            assert p(arch, T, out_f32=False) is None, (arch, T)
    assert p("sm_100a", 6144) is None
    assert p("sm_103a", 6144) == (64, 2)

    sms_list = (148, 150, 151, 152, 153, 160)
    near = (-64, -1, 0, 1, 64)
    Ts = {T + d for T in K1_152_TWO_SPLIT_MEASURED_TOKENS for d in near}
    Ts |= {16172, 16231, 16384, 65536}

    def plans():
        return {
            (arch, sms, T, f32): p(arch, T, sm_count=sms, out_f32=f32)
            for arch, _ in K1_152_TWO_SPLIT_RANGES
            for sms in sms_list
            for T in Ts
            for f32 in (False, True)
        }

    with_table = plans()
    monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
    without = plans()
    routed = 0
    for (arch, sms, T, f32), plan in with_table.items():
        assert without[(arch, sms, T, f32)] is None, (arch, sms, T, f32)
        held = [
            tuple(v)
            for lo, hi, v in K1_152_TWO_SPLIT_RANGES.get((arch, sms), ())
            if lo <= T <= hi
        ]
        if f32 and held:
            assert plan == held[0], (arch, sms, T)
            routed += 1
        else:
            assert plan is None, (arch, sms, T, f32)
    assert routed


#: The ranges of :data:`K1_ONE_SPLIT_RANGES_BY_SM`, ``(arch, sm_count) -> {(d_in, d_out): ((min_tokens, max_tokens,
#: block_n), ...)}`` (FP32 out): one split of 64-wide tiles at every token count of a range, at both block parities.
K1_152_ONE_SPLIT_RANGES = {
    ("sm_100a", 152): {(4096, 32): ((256, 2048, 64),), (4096, 128): ((256, 3328, 64),)},
    ("sm_103a", 152): {(4096, 32): ((256, 2048, 64),), (4096, 128): ((256, 2816, 64),)},
}
#: The token counts at which these rows were measured on 152 SMs (GB200, GB300; 3328 to 3840 on sm_100a only).
# fmt: off
K1_152_ONE_SPLIT_MEASURED_TOKENS = (
    256, 384, 512, 768, 800, 1024, 1025, 1536, 1600, 2048, 2049, 2304, 2560, 2561, 2816, 3072, 3073, 3328, 3584,
    3585, 3840, 4096,
)
# fmt: on


def test_k1_152_sm_one_split_ranges(monkeypatch):
    """On 152-SM sm_100a and sm_103a parts the FP32 rows ``4096 -> 32`` and ``4096 -> 128`` run one split of 64-wide
    tiles of one accumulation chain at every token count of their ranges in :data:`K1_ONE_SPLIT_RANGES_BY_SM`, at an
    even and at an odd number of 64-token K blocks, in place of the routing of the other tables there (two splits, the
    deterministic program at excluded token counts, and ``4096 -> 128`` outside its two-split range).  One token below
    or above a range, BF16 out, every other row and every other SM count (148, 150, 151, 153,
    160) keep the route they take without the table.  A range starts and ends at a measured token count and lies below
    the window of :data:`K1_SPLIT_RULES_BY_SM_TOKENS`; no chain range of :data:`K1_CHAIN_RANGES_BY_SM` covers it.
    The table has no other part (148-SM sm_100a ``4096 -> 1024`` BF16 runs two splits in the window instead,
    :data:`K1_ORDERED_RANGES_BY_SM`, ``test_k1_wgrad_changed_rows_take_their_route``)."""
    table = {
        part: rows
        for part, rows in cake_backend.K1_ONE_SPLIT_RANGES_BY_SM.items()
        if part[1] == 152
    }
    assert len(cake_backend.K1_ONE_SPLIT_RANGES_BY_SM) == len(table)
    first, _ = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    assert table == {
        part: {(part[0], 1, M, N, "f32"): want for (M, N), want in rows.items()}
        for part, rows in K1_152_ONE_SPLIT_RANGES.items()
    }
    for (arch, sms), rows in table.items():
        assert not cake_backend.K1_CHAIN_RANGES_BY_SM.get((arch, sms))
        for key, ranges in rows.items():
            for lo, hi, block_n in ranges:
                assert lo in K1_152_ONE_SPLIT_MEASURED_TOKENS, key
                assert hi in K1_152_ONE_SPLIT_MEASURED_TOKENS, key
                assert block_n in cake_backend.K1_BLOCK_N_CHOICES, key
                assert block_n not in cake_backend.K1_ROW_MAJOR_BLOCK_N, key
                assert lo <= hi < first, key

    def p(arch, N, T, sm_count=152, out_f32=True, M=4096):
        return cake_backend.k1_plan(
            arch=arch, L=1, M=M, N=N, K=T, out_f32=out_f32, sm_count=sm_count
        )

    for (arch, sms), rows in K1_152_ONE_SPLIT_RANGES.items():
        for (M, N), ranges in rows.items():
            for lo, hi, block_n in ranges:
                inside = {lo, lo + 1, 511, 512, 800, 1000, 1024, 1025, 1600, 2047, 2048}
                inside |= {hi - 1, hi}
                for T in sorted(t for t in inside if lo <= t <= hi):
                    assert p(arch, N, T) == (block_n, 1), (arch, N, T)
                    kw = dict(arch=arch, L=1, M=M, N=N, K=T, out_f32=True)
                    assert cake_backend.k1_chain_plan(sm_count=sms, **kw) is None
    sms_list = (148, 150, 151, 152, 153, 160)
    near = (-1, 0, 1)
    Ts = {t + d for t in K1_152_ONE_SPLIT_MEASURED_TOKENS for d in near}
    Ts |= {255, 1000, 16231}
    shapes = ((4096, 32), (4096, 128), (4096, 48), (5120, 32))

    def plans():
        return {
            (arch, sms, M, N, T, f32): p(arch, N, T, sm_count=sms, out_f32=f32, M=M)
            for arch in ("sm_100a", "sm_103a")
            for sms in sms_list
            for M, N in shapes
            for T in Ts
            for f32 in (False, True)
        }

    with_table = plans()
    monkeypatch.setattr(cake_backend, "K1_ONE_SPLIT_RANGES_BY_SM", {})
    without = plans()
    changed = 0
    for (arch, sms, M, N, T, f32), plan in with_table.items():
        own = K1_152_ONE_SPLIT_RANGES.get((arch, sms), {}).get((M, N), ())
        held = [bn for lo, hi, bn in own if lo <= T <= hi]
        if f32 and held:
            assert plan == (held[0], 1), (arch, sms, M, N, T)
            changed += plan != without[(arch, sms, M, N, T, f32)]
        else:
            assert plan == without[(arch, sms, M, N, T, f32)], (arch, sms, M, N, T, f32)
    assert changed


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize(
    "d_out, T", [(32, 256), (32, 1025), (32, 2048), (128, 800), (128, 2049)]
)
def test_k1_152_sm_one_split_ranges_plan_the_measured_launch(
    monkeypatch, arch, d_out, T
):
    """The planner's launch at a token count of a one-split range is the explicit ``splits=1, block_n=64`` launch the
    range was measured with: the transposed-store program ``dense_proj_gemm_atomic_nn_n64_f32_t`` of one chain over 16
    (``4096 -> 32``) or 32 (``4096 -> 128``) tiles of 256 x 64, one CTA pair per tile (76 pairs on 152 SMs), with no
    workspace and no tickets.  On 148 SMs the row does not run it.  The cluster ranges of :data:`K1_DSM_RANGES_BY_SM`,
    which come first and hold some of these token counts on sm_100a, are taken out."""
    _register_k1(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    G = torch.zeros(T, d_out, dtype=torch.bfloat16)
    X = torch.zeros(T, 4096, dtype=torch.bfloat16)
    out = torch.zeros(d_out, 4096, dtype=torch.float32)
    kw = dict(transposed_out=True, arch=arch, sm_count=152)
    plan, *_, reason = cake_backend.plan_atomic_gemm(X.t(), G, out, **kw)
    assert (
        isinstance(plan, cake_backend.CakeDenseGemmNondetProjectionPlan)
        and reason == ""
    )
    explicit = cake_backend.plan_atomic_gemm(X.t(), G, out, splits=1, block_n=64, **kw)
    assert plan == explicit[0]
    assert (plan.template, plan.block_n, plan.splits, plan.chains, plan.out_t) == (
        "dense_proj_gemm_atomic_nn_n64_f32_t",
        64,
        1,
        1,
        True,
    )
    assert (plan.M, plan.N, plan.K, plan.m_tiles, plan.n_tiles) == (
        4096,
        d_out,
        T,
        16,
        -(-d_out // 64),
    )
    assert plan.grid == (2 * plan.tiles, 1, 1) and plan.sm_pairs == 76
    assert plan.ws_f32_elems == 0 and plan.counters_u32 == 0
    other = cake_backend.plan_atomic_gemm(
        X.t(), G, out, transposed_out=True, arch=arch, sm_count=148
    )[0]
    assert other is None or (other.block_n, other.splits) != (64, 1)


def test_k1_152_sm_one_split_ranges_bind_the_one_split_program(monkeypatch):
    """The weight gradient of a row at a token count of a one-split range (152-SM sm_103a
    part, ``4096 -> 128`` FP32 at 1000 tokens) binds this package's one-split program in ``_K1_ATOMIC_CACHE``: no
    arena (one split: no workspace, no tickets), no deterministic plan, one launch per call."""
    table = cake_backend.K1_ONE_SPLIT_RANGES_BY_SM
    arch = "sm_103a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "K1_ONE_SPLIT_RANGES_BY_SM", table)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16, T = torch.bfloat16, 1000
    X, G = torch.empty(T, 4096, dtype=bf16), torch.empty(T, 128, dtype=bf16)
    out_w = torch.empty(128, 4096)
    for _ in range(2):
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w)
            is out_w
        )
    assert len(entry_fn.calls) == 2
    ((_, atomic),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert atomic.plan is not None and atomic.k1 is not None
    assert (atomic.plan.block_n, atomic.plan.splits, atomic.plan.chains) == (64, 1, 1)
    assert atomic.plan.template == "dense_proj_gemm_atomic_nn_n64_f32_t"
    assert atomic.arena is None and atomic.det_plan is None


#: The ranges of :data:`K1_DSM_RANGES_BY_SM`, ``(arch, sm_count) -> {(d_in, d_out): ((min_tokens, max_tokens, (block_n,
#: splits)), ...)}`` (FP32 out): the cluster program of that tile width and cluster size at every token count of a range.
# fmt: off
K1_DSM_RANGES = {
    ("sm_100a", 148): {
        (4096, 32): ((1024, 32768, (32, 2)),),
        (4096, 48): ((1024, 4095, (32, 2)), (4096, 16171, (64, 2)), (16172, 16384, (32, 2)), (32768, 32768, (64, 2))),
        (4096, 64): (
            (1024, 4095, (32, 2)), (4096, 6144, (64, 2)), (12288, 16171, (64, 2)), (16172, 16384, (32, 2)),
            (16385, 32768, (64, 2)),
        ),
        (4096, 128): ((1024, 12288, (64, 2)),),
        (5120, 32): ((1024, 32768, (32, 2)),),
        (5120, 48): ((1024, 32768, (64, 2)),),
        (5120, 64): ((1024, 2047, (64, 3)), (2048, 32768, (64, 2))),
        (5120, 128): ((1024, 4095, (128, 2)), (4096, 32768, (128, 3))),
    },
    ("sm_100a", 152): {
        (4096, 32): ((1024, 2048, (32, 2)), (6144, 32768, (32, 2))),
        (4096, 48): ((1024, 2048, (64, 2)), (4096, 8192, (64, 2)), (16172, 16384, (32, 2)), (16385, 32768, (64, 2))),
        (4096, 64): ((1024, 16171, (64, 2)), (16172, 16384, (32, 2)), (16385, 32768, (64, 2))),
        (4096, 128): ((4096, 6144, (64, 2)), (6145, 8192, (128, 2))),
        (5120, 32): ((1024, 32768, (32, 2)),),
        (5120, 48): ((1024, 24576, (64, 2)),),
        (5120, 64): ((1024, 2048, (64, 2)), (4096, 24576, (64, 2))),
        (5120, 128): ((1024, 6144, (128, 2)),),
    },
    ("sm_103a", 152): {
        (4096, 32): ((1024, 32768, (32, 2)),),
        (4096, 48): (
            (1024, 4095, (32, 2)), (4096, 6144, (64, 2)), (12288, 16171, (64, 2)), (16172, 16384, (32, 2)),
            (16385, 32768, (64, 2)),
        ),
        (4096, 64): (
            (1024, 2047, (32, 2)), (2048, 8191, (64, 2)), (8192, 12287, (32, 2)), (12288, 16171, (64, 2)),
            (16172, 16384, (32, 2)), (16385, 32768, (64, 2)),
        ),
        (4096, 128): ((1024, 12288, (64, 2)),),
        (5120, 32): ((1024, 2048, (32, 2)), (6144, 8191, (32, 2)), (12288, 16383, (32, 2)), (16385, 32768, (32, 2))),
        (5120, 48): ((2048, 4095, (64, 2)),),
        (5120, 64): ((1024, 4095, (64, 2)), (6144, 12287, (64, 2))),
        (5120, 128): ((1024, 4095, (128, 2)),),
    },
    ("sm_107a", 212): {
        (4096, 48): ((16172, 16384, (64, 4)),),
        (4096, 64): ((16172, 16384, (64, 4)),),
    },
}
#: The token counts at which the rows of each part were measured (two 148-SM B200 parts, two 152-SM GB200 parts, two
#: 152-SM GB300 parts; also 16172 and 16231 tokens, where ``4096 -> 48`` and ``4096 -> 64`` were measured for the
#: parity ranges of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` on one B200, one GB200 and one GB300 part; the 212-SM R200
#: rows at 16172, 16231 and 16384 tokens).
K1_DSM_MEASURED_TOKENS = {
    ("sm_100a", 148): (1024, 2048, 4096, 6144, 8192, 12288, 16172, 16231, 16384, 24576, 32768),
    ("sm_100a", 152): (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16172, 16231, 16384, 24576, 32768),
    ("sm_103a", 152): (1024, 2048, 4096, 6144, 8192, 12288, 16172, 16231, 16384, 24576, 32768),
    ("sm_107a", 212): (16172, 16231, 16384),
}
# fmt: on


def _k1_dsm_ranges_edges(ranges, part):
    """Token counts of a row's cluster ranges to plan at: every range's first and last token count and the token counts
    next to them, and the token counts at which the rows of ``part`` were measured."""
    Ts = {t for lo, hi, _ in ranges for t in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1)}
    return sorted(Ts | set(K1_DSM_MEASURED_TOKENS[part]))


def _k1_wgrad_plans(M, N, T, parts):
    """``{(arch, sm_count, f32): (template, block_n, splits, chains, dsm) or None}``: the planner's weight-gradient
    launch ``dW[N, M] = G[T, N].T @ X[T, M]`` on each part of ``parts`` for BF16 and FP32 out (CPU tensors, never
    written: nothing is launched); ``None``: the deterministic program."""
    G, X = (
        torch.empty(T, N, dtype=torch.bfloat16),
        torch.empty(T, M, dtype=torch.bfloat16),
    )
    plans = {}
    for f32 in (False, True):
        out = torch.empty(N, M, dtype=torch.float32 if f32 else torch.bfloat16)
        for arch, sm_count in parts:
            plan = cake_backend.plan_atomic_gemm(
                X.t(), G, out, transposed_out=True, arch=arch, sm_count=sm_count
            )[0]
            plans[(arch, sm_count, f32)] = (
                None
                if plan is None
                else (plan.template, plan.block_n, plan.splits, plan.chains, plan.dsm)
            )
    return plans


def test_k1_dsm_ranges(monkeypatch):
    """On 148-SM and 152-SM sm_100a parts and on 152-SM sm_103a parts the FP32 rows ``4096 -> 32, 48, 64, 128`` and
    ``5120 -> 32, 48, 64, 128``, and on 212-SM sm_107a parts ``4096 -> 48, 64``, run the cluster program of their
    ranges in :data:`K1_DSM_RANGES_BY_SM`
    (:func:`k1_dsm_plan`; :func:`plan_atomic_gemm` plans it with ``dsm``, before every other K1 table) at every token
    count of a range.  A row's ranges are ordered and disjoint, its first range starts and its last range ends at a
    token count at which the part was measured, every range holds such a token count and ends at one or next to one
    (where the next range starts), and every range names a program this checkout registers for the part's
    architecture, with its cluster size baked into the module.  One token below or above a row's ranges, BF16 out,
    every other row and every other part (SM count or architecture) keep the plan they take without the part's table
    (another part's table where it has one); every row's plan changes inside each of its ranges.  The odd block counts
    of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`, which the planner reads inside a cluster range, are taken out of the
    plan comparison (``test_k1_dsm_odd_block_ranges`` holds them)."""
    table = cake_backend.K1_DSM_RANGES_BY_SM
    assert table == {
        part: {(part[0], 1, M, N, "f32"): want for (M, N), want in rows.items()}
        for part, rows in K1_DSM_RANGES.items()
    }
    for part, rows in table.items():
        arch = part[0]
        measured = set(K1_DSM_MEASURED_TOKENS[part])
        for key, ranges in rows.items():
            # a range may also end next to a token count the part runs on the deterministic program (its exact
            # exclusions, such as the former three-CTA ranges of sm_103a 5120 -> x, or the part's entries of the
            # exclusions of every SM count)
            det = (
                *cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM.get(part, {}).get(
                    key, ()
                ),
                *(
                    (lo, hi)
                    for sms, lo, hi in cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS.get(
                        key, ()
                    )
                    if sms == part[1]
                ),
            )
            det_ts = {
                t
                for lo, hi, _ in ranges
                for t in (lo - 1, hi + 1)
                if any(a <= t and (b is None or t <= b) for a, b in det)
            }
            assert ranges[0][0] in measured or ranges[0][0] - 1 in det_ts, key
            assert ranges[-1][1] in measured or ranges[-1][1] + 1 in det_ts, key
            for i, (lo, hi, (block_n, splits)) in enumerate(ranges):
                assert lo <= hi and any(lo <= t <= hi for t in measured), key
                nxt = ranges[i + 1][0] if i + 1 < len(ranges) else None
                prv = ranges[i - 1][1] if i else None
                assert hi in measured or hi + 1 == nxt or hi + 1 in det_ts, key
                assert lo in measured or lo - 1 == prv or lo - 1 in det_ts, key
                if i:
                    assert ranges[i - 1][1] < lo, key
                assert cake_backend.k1_splits_fit(splits, lo), key
                template = cake_backend.dsm_k1_template(block_n, splits)
                name = cake_jit.KERNELS[arch][template]
                assert cake_jit.MODULES[name]["launch"]["cluster"] == [splits, 1, 1]
    _register_k1(monkeypatch, "sm_100a")
    _register_k1(monkeypatch, "sm_103a")
    _register_k1(monkeypatch, "sm_107a")

    def p(arch, sm_count, M, N, T, out_f32=True):
        return cake_backend.k1_dsm_plan(
            arch=arch, L=1, M=M, N=N, K=T, out_f32=out_f32, sm_count=sm_count
        )

    def held_by(part, M, N, T):
        own = K1_DSM_RANGES.get(part, {}).get((M, N), ())
        held = [v for lo, hi, v in own if lo <= T <= hi]
        return held[0] if held else None

    others = [
        (a, s)
        for a in ("sm_100a", "sm_103a")
        for s in (147, 148, 150, 151, 152, 153, 160)
    ]
    others += [("sm_107a", 152), ("sm_107a", 212), ("sm_107a", 216)]
    for part, rows in K1_DSM_RANGES.items():
        arch, sms = part
        for (M, N), ranges in rows.items():
            for T in _k1_dsm_ranges_edges(ranges, part):
                assert p(arch, sms, M, N, T) == held_by(part, M, N, T), (part, M, N, T)
                assert p(arch, sms, M, N, T, out_f32=False) is None, (part, M, N, T)
                for other in others:
                    if other != part:
                        want = held_by(other, M, N, T)
                        assert p(*other, M, N, T) == want, (part, other, M, N, T)
    shapes = sorted({mn for rows in K1_DSM_RANGES.values() for mn in rows})
    shapes += [(4096, 512), (6144, 32), (6144, 128), (5120, 1024)]
    Ts = {255, 256, 512, 960, 1023, 16231, 40960}
    for part, rows in K1_DSM_RANGES.items():
        Ts |= set(K1_DSM_MEASURED_TOKENS[part])
        for ranges in rows.values():
            Ts |= set(_k1_dsm_ranges_edges(ranges, part))
    parts = [("sm_100a", s) for s in (148, 150, 151, 152, 153, 160)]
    parts += [("sm_103a", 148), ("sm_103a", 152), ("sm_107a", 212), ("sm_107a", 216)]

    def plans():
        return {
            (arch, sms, M, N, T, f32): plan
            for M, N in shapes
            for T in sorted(Ts)
            for (arch, sms, f32), plan in _k1_wgrad_plans(M, N, T, parts).items()
        }

    monkeypatch.setattr(cake_backend, "K1_DSM_ODD_BLOCK_RANGES_BY_SM", {})
    with_table = plans()
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    without = plans()
    changed = {}
    for (arch, sms, M, N, T, f32), plan in with_table.items():
        own = K1_DSM_RANGES.get((arch, sms), {}).get((M, N), ())
        held = [(i, v) for i, (lo, hi, v) in enumerate(own) if lo <= T <= hi]
        if f32 and held:
            i, (block_n, splits) = held[0]
            want = (
                cake_backend.dsm_k1_template(block_n, splits),
                block_n,
                splits,
                1,
                True,
            )
            assert plan == want, (arch, sms, M, N, T)
            assert plan != without[(arch, sms, M, N, T, f32)], (arch, sms, M, N, T)
            changed[(arch, sms, M, N, i)] = changed.get((arch, sms, M, N, i), 0) + 1
        else:
            assert plan == without[(arch, sms, M, N, T, f32)], (arch, sms, M, N, T, f32)
            assert plan is None or not plan[4], (arch, sms, M, N, T, f32)
    assert sorted(changed) == sorted(
        (*part, M, N, i)
        for part, rows in K1_DSM_RANGES.items()
        for (M, N), ranges in rows.items()
        for i in range(len(ranges))
    )


#: The ranges of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`, ``(arch, sm_count) -> {(d_in, d_out): ((min_tokens,
#: max_tokens, program), ...)}`` (FP32 out; ``program`` ``None`` for the deterministic program, ``(block_n, splits)``
#: for the cluster program of that tile, else ``(block_n, splits, chains, order)`` of the CTA-pair program).
# fmt: off
K1_DSM_ODD_BLOCK_RANGES = {
    ("sm_100a", 148): {(4096, 48): ((16172, 16384, (64, 4)),), (4096, 64): ((16172, 16384, (64, 4)),)},
    ("sm_100a", 152): {(4096, 48): ((16172, 16384, (64, 4)),), (4096, 64): ((16172, 16384, (64, 4)),)},
    ("sm_103a", 152): {(4096, 48): ((16172, 16384, (64, 4)),), (4096, 64): ((16172, 16384, (64, 4)),)},
}
# fmt: on


def _assert_k1_odd_block_range(arch, sms, M, N, lo, hi, program):
    """One range of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`: it lies inside one cluster range of the row, of two CTAs
    of 32-wide tiles, and names a registered program: the cluster program of a tile of three or more CTAs, a
    three-or-more-split CTA-pair program, or the deterministic program.  At every token count from ``lo`` to ``hi``
    the planner's FP32 launch is that program at an odd number of 64-token K blocks -- never a two-way split -- and
    the cluster program of the cluster range at an even one; one token outside the range, BF16 out and another SM
    count get no program from the table."""
    ((cluster,),) = [
        [v for a, b, v in K1_DSM_RANGES[(arch, sms)][(M, N)] if a <= lo <= hi <= b]
    ]
    assert cluster == (32, 2), (arch, sms, M, N)
    if program is not None and len(program) == 2:
        block_n, splits = program
        assert splits > 2
        template = cake_backend.dsm_k1_template(block_n, splits)
        assert template in cake_jit.KERNELS[arch], template
        name = cake_jit.KERNELS[arch][template]
        assert cake_jit.MODULES[name]["launch"]["cluster"] == [splits, 1, 1]
    elif program is not None:
        block_n, splits, chains, order = program
        assert order in cake_backend.K1_SPLIT_ORDERS and splits > 2
        template = cake_backend.atomic_k1_template(block_n, True, chains, order)
        assert template in cake_jit.KERNELS[arch], template
    G = torch.empty(hi + 1, N, dtype=torch.bfloat16)
    X = torch.empty(hi + 1, M, dtype=torch.bfloat16)
    out = torch.empty(N, M, dtype=torch.float32)
    for T in range(lo - 1, hi + 2):
        where = (arch, sms, M, N, T)
        kw = dict(arch=arch, L=1, M=M, N=N, K=T, out_f32=True, sm_count=sms)
        held = -(-T // 64) % 2 == 1 and lo <= T <= hi
        got = cake_backend.k1_dsm_odd_block_plan(**kw)
        assert got == ((program or ()) if held else None), where
        for other in (dict(kw, out_f32=False), dict(kw, sm_count=sms + 1)):
            assert cake_backend.k1_dsm_odd_block_plan(**other) is None, where
        if not lo <= T <= hi:
            continue
        plan = cake_backend.plan_atomic_gemm(
            X[:T].t(), G[:T], out, transposed_out=True, arch=arch, sm_count=sms
        )[0]
        if not held:
            assert plan is not None and plan.dsm, where
            assert (plan.block_n, plan.splits) == (32, 2), where
        elif program is None:
            assert plan is None, where
        elif len(program) == 2:
            assert plan.dsm and plan.reproducible, where
            assert (plan.block_n, plan.splits) == program, where
            assert plan.template == cake_backend.dsm_k1_template(*program), where
        else:
            assert not plan.dsm and not plan.reproducible and plan.splits > 2, where
            got = (plan.block_n, plan.splits, plan.chains, plan.split_order)
            assert got == program, where


def test_k1_dsm_odd_block_ranges(monkeypatch):
    """On 148-SM and 152-SM sm_100a parts and 152-SM sm_103a parts the FP32 rows ``4096 -> 48`` and ``4096 -> 64`` run
    the program of their range in :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` (:func:`k1_dsm_odd_block_plan`) at every token
    count of the range that spans an odd number of 64-token K blocks: four CTAs of the cluster program over 64-wide
    tiles (their partials added in cluster-rank order: the same bits on every run).  At an even block count they run
    two CTAs of 32-wide tiles, the cluster range that holds the range.  At no token count of a range with an odd block
    count does the planner give a two-way split (every two-way split missed the accuracy check there).  BF16 out, a
    token count outside the ranges, the other rows and the other parts keep the plan they take without the table, and
    without the cluster ranges the table does not apply."""
    table = cake_backend.K1_DSM_ODD_BLOCK_RANGES_BY_SM
    assert table == {
        part: {(part[0], 1, M, N, "f32"): want for (M, N), want in rows.items()}
        for part, rows in K1_DSM_ODD_BLOCK_RANGES.items()
    }
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        _register_k1(monkeypatch, arch)
    for (arch, sms), rows in K1_DSM_ODD_BLOCK_RANGES.items():
        for (M, N), ranges in rows.items():
            for lo, hi, program in ranges:
                _assert_k1_odd_block_range(arch, sms, M, N, lo, hi, program)
    parts = [("sm_100a", s) for s in (147, 148, 150, 152, 153)]
    parts += [("sm_103a", 148), ("sm_103a", 152), ("sm_107a", 212)]
    Ts = (16171, 16172, 16173, 16192, 16193, 16231, 16256, 16257, 16300, 16320)
    Ts += (16321, 16384, 16385)
    shapes = ((4096, 32), (4096, 48), (4096, 64), (4096, 128), (5120, 48), (5120, 64))

    def plans():
        return {
            (arch, sms, M, N, T, f32): plan
            for M, N in shapes
            for T in Ts
            for (arch, sms, f32), plan in _k1_wgrad_plans(M, N, T, parts).items()
        }

    with_table = plans()
    monkeypatch.setattr(cake_backend, "K1_DSM_ODD_BLOCK_RANGES_BY_SM", {})
    without = plans()
    changed = set()
    for (arch, sms, M, N, T, f32), plan in with_table.items():
        own = K1_DSM_ODD_BLOCK_RANGES.get((arch, sms), {}).get((M, N), ())
        held = [v for lo, hi, v in own if lo <= T <= hi and -(-T // 64) % 2 == 1]
        if f32 and held:
            assert plan != without[(arch, sms, M, N, T, f32)], (arch, sms, M, N, T)
            changed.add((arch, sms, M, N))
        else:
            assert plan == without[(arch, sms, M, N, T, f32)], (arch, sms, M, N, T, f32)
    rows = K1_DSM_ODD_BLOCK_RANGES
    assert changed == {(*part, M, N) for part in rows for M, N in rows[part]}
    # without the cluster ranges the odd-block table does not apply
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    neither = plans()
    monkeypatch.setattr(cake_backend, "K1_DSM_ODD_BLOCK_RANGES_BY_SM", table)
    assert plans() == neither


@pytest.mark.parametrize(
    "d_in, d_out, T",
    [
        (4096, 32, 1024),
        (4096, 32, 2048),
        (4096, 32, 32768),
        (4096, 48, 16384),
        (4096, 128, 4096),
        (4096, 128, 6145),
        (4096, 128, 8192),
        (5120, 48, 24576),
        (5120, 128, 6144),
        (5120, 64, 1025),
        (5120, 64, 4096),
    ],
)
def test_k1_152_sm_dsm_ranges_plan_the_cluster_launch(monkeypatch, d_in, d_out, T):
    """The planner's launch at a token count of a cluster range of a 152-SM sm_100a part is the range's cluster
    program (:func:`_assert_dsm_cluster_launch`)."""
    _assert_dsm_cluster_launch(monkeypatch, ("sm_100a", 152), d_in, d_out, T)


@pytest.mark.parametrize(
    "part, d_in, d_out, T",
    [
        (("sm_103a", 152), 4096, 32, 1024),
        (("sm_103a", 152), 4096, 32, 2048),
        (("sm_103a", 152), 4096, 64, 8192),
        (("sm_103a", 152), 4096, 128, 12288),
        (("sm_103a", 152), 5120, 32, 6144),
        (("sm_103a", 152), 5120, 48, 2048),
        (("sm_103a", 152), 5120, 64, 6144),
        (("sm_103a", 152), 5120, 128, 2048),
        (("sm_100a", 148), 4096, 32, 4096),
        (("sm_100a", 148), 4096, 48, 32768),
        (("sm_100a", 148), 4096, 64, 12288),
        (("sm_100a", 148), 4096, 128, 1024),
        (("sm_100a", 148), 5120, 48, 2048),
        (("sm_100a", 148), 5120, 64, 1024),
        (("sm_100a", 148), 5120, 128, 4096),
    ],
)
def test_k1_dsm_ranges_plan_the_cluster_launch_on_more_parts(
    monkeypatch, part, d_in, d_out, T
):
    """The planner's launch at a token count of a cluster range of a 152-SM sm_103a or a 148-SM sm_100a part is the
    range's cluster program (:func:`_assert_dsm_cluster_launch`)."""
    _assert_dsm_cluster_launch(monkeypatch, part, d_in, d_out, T)


def _assert_dsm_cluster_launch(monkeypatch, part, d_in, d_out, T):
    """The planner's launch on ``part`` at a token count of a cluster range is the range's cluster program: template
    ``dense_proj_gemm_dsm_nn_n<block_n>_f32_t_s<splits>``, one chain, the transposed store, tiles of 128 ``d_in``
    columns by ``block_n`` ``d_out`` rows, ``splits`` CTAs per tile (a grid that is a multiple of the module's
    cluster), no workspace and no tickets; the program is registered for sm_100a and sm_103a.  An explicit split count
    or tile width plans the CTA-pair program.  On another part the row runs the cluster program of that part's own
    range at ``T`` and no cluster program where that part has none."""
    arch, sms = part
    _register_k1(monkeypatch, arch)
    G = torch.zeros(T, d_out, dtype=torch.bfloat16)
    X = torch.zeros(T, d_in, dtype=torch.bfloat16)
    out = torch.zeros(d_out, d_in, dtype=torch.float32)
    kw = dict(transposed_out=True, arch=arch, sm_count=sms)
    plan, a, b, out3, reason = cake_backend.plan_atomic_gemm(X.t(), G, out, **kw)
    ((block_n, splits),) = [
        v for lo, hi, v in K1_DSM_RANGES[part][(d_in, d_out)] if lo <= T <= hi
    ]
    assert (
        isinstance(plan, cake_backend.CakeDenseGemmNondetProjectionPlan)
        and reason == ""
    )
    assert plan.dsm and not plan.swapped and not plan.workspace
    assert (plan.template, plan.block_n, plan.splits, plan.chains, plan.out_t) == (
        f"dense_proj_gemm_dsm_nn_n{block_n}_f32_t_s{splits}",
        block_n,
        splits,
        1,
        True,
    )
    assert (plan.M, plan.N, plan.K, plan.m_tiles, plan.n_tiles) == (
        d_in,
        d_out,
        T,
        -(-d_in // 128),
        -(-d_out // block_n),
    )
    assert (
        plan.grid == (plan.tiles * splits, 1, 1) and plan.items == plan.tiles * splits
    )
    assert plan.ws_f32_elems == 0 and plan.counters_u32 == 0
    record = cake_jit.MODULES[cake_jit.KERNELS[arch][plan.template]]
    assert plan.grid[0] % record["launch"]["cluster"][0] == 0
    assert record["arches"] == ["sm_100a", "sm_103a"]
    assert tuple(a.shape) == (1, T, d_in) and tuple(b.shape) == (1, T, d_out)
    assert tuple(out3.shape) == (1, d_out, d_in)
    explicit = cake_backend.plan_atomic_gemm(X.t(), G, out, splits=2, **kw)[0]
    assert explicit is not None and not explicit.dsm
    assert explicit.template.startswith(cake_jit.TEMPLATE_PREFIXES[0])
    explicit = cake_backend.plan_atomic_gemm(X.t(), G, out, block_n=64, **kw)[0]
    assert explicit is not None and not explicit.dsm
    others = [(a_, s_) for a_ in ("sm_100a", "sm_103a") for s_ in (148, 150, 152)]
    for other_arch, other_sms in others:
        if (other_arch, other_sms) == part:
            continue
        _register_k1(monkeypatch, other_arch)
        other = cake_backend.plan_atomic_gemm(
            X.t(), G, out, transposed_out=True, arch=other_arch, sm_count=other_sms
        )[0]
        own = K1_DSM_RANGES.get((other_arch, other_sms), {}).get((d_in, d_out), ())
        held = [v for lo, hi, v in own if lo <= T <= hi]
        if held:
            assert other is not None and other.dsm, (other_arch, other_sms)
            assert (other.block_n, other.splits) == held[0], (other_arch, other_sms)
        else:
            assert other is None or not other.dsm, (other_arch, other_sms)
    # a checkout that does not register the program: the deterministic program, with the reason
    table = dict(cake_jit.KERNELS[arch])
    table.pop(plan.template)
    monkeypatch.setitem(cake_jit.KERNELS, arch, table)
    missing = cake_backend.plan_atomic_gemm(X.t(), G, out, **kw)
    assert missing[0] is None and plan.template in missing[4]


def test_k1_152_sm_dsm_ranges_bind_the_cluster_program(monkeypatch):
    """The weight gradient of a row at a token count of a cluster range (152-SM sm_100a
    part, ``4096 -> 64`` FP32 at 2048 tokens) binds this package's cluster program in ``_K1_ATOMIC_CACHE``: no arena
    (no workspace, no tickets), no deterministic plan, one launch per call whose arguments are the program's argument
    plan (the operand views, ``out``, the extents, the output strides, the PDL flag and the plan's grid of
    ``splits`` CTAs per tile).  The prepared launch binds the same program and grid."""
    table = cake_backend.K1_DSM_RANGES_BY_SM
    arch = "sm_100a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", table)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16, T = torch.bfloat16, 2048
    X, G = torch.empty(T, 4096, dtype=bf16), torch.empty(T, 64, dtype=bf16)
    out_w = torch.empty(64, 4096)
    for _ in range(2):
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w)
            is out_w
        )
    assert len(entry_fn.calls) == 2
    ((_, atomic),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert atomic.plan is not None and atomic.k1 is not None and atomic.plan.dsm
    assert (atomic.plan.block_n, atomic.plan.splits, atomic.plan.chains) == (64, 2, 1)
    assert atomic.plan.template == "dense_proj_gemm_dsm_nn_n64_f32_t_s2"
    assert atomic.module_name == cake_jit.KERNELS[arch][atomic.plan.template]
    assert atomic.arena is None and atomic.det_plan is None
    arg_plan = cake_jit.arg_plan_of(cake_jit.MODULES[atomic.module_name])
    names = [name for _, name in arg_plan]
    assert "ws" not in names and "counters" not in names and "num_splits" not in names
    (args, _) = entry_fn.calls[-1]
    assert len(args) == len(arg_plan)
    values = dict(zip(names, args, strict=True))
    assert (values["M"], values["N"], values["K"], values["num_l"]) == (4096, 64, T, 1)
    assert (values["ldo"], values["out_l"]) == (4096, 64 * 4096)
    assert values["out"].data_ptr() == out_w.data_ptr()
    tiles = 32 * 1
    assert (values["grid_x"], values["grid_y"], values["grid_z"]) == (tiles * 2, 1, 1)
    prepared = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
        G, X, out_w
    )
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
    assert prepared.plan == atomic.plan and prepared.module_name == atomic.module_name
    assert prepared.grid == (tiles * 2, 1, 1) and prepared.counters_are_zero()
    prepared.reset_counters()  # no tickets: nothing to zero


#: The range of :data:`K1_TWO_SPLIT_RANGES_BY_SM` of the FP32 row ``4096 -> 128`` on 212-SM sm_107a parts (no
#: verdict in :data:`K1_SPLIT_RULES` or in the part's table of :data:`K1_SPLIT_RULES_BY_SM`): two splits of 64-wide
#: tiles at every token count of the range (either number of 64-token K blocks).
K1_212_4096_128_RANGE = ((1024, 7680, (64, 2)),)
#: The measured token counts of that row on 212 SMs.
K1_212_N128_TOKENS = (512, 768, 896, 1024, 2048, 4096, 6144, 6656, 7168, 7680)


def test_k1_212_sm_4096_to_128_two_split_range(monkeypatch):
    """On 212-SM sm_107a parts (below the SM-count floor) the FP32 row ``4096 -> 128``, which has no verdict in
    :data:`K1_SPLIT_RULES` or in the part's table of :data:`K1_SPLIT_RULES_BY_SM`, runs two splits of 64-wide tiles at
    every token count from 1024 to 7680, at an even or an odd number of 64-token K blocks (1025, 1088, 3000, 4160, 5000,
    7615).  A token count outside the range (512, 896, 1023, 7681, 7745, 8192, 16384), BF16, and every other SM count
    (208, 210, 211, 213, 216, 217) run the deterministic program as without the table."""
    key = ("sm_107a", 1, 4096, 128, "f32")
    part = ("sm_107a", 212)
    two = cake_backend.K1_TWO_SPLIT_RANGES_BY_SM[part]
    assert two[key] == K1_212_4096_128_RANGE
    assert cake_backend.K1_SPLIT_RULES[key] is None
    assert key not in cake_backend.K1_SPLIT_RULES_BY_SM[part]
    assert key not in cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES.get(part, {})
    assert key not in cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM.get(part, {})
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    for lo, hi, choice in K1_212_4096_128_RANGE:
        assert lo in K1_212_N128_TOKENS, key
        assert hi in K1_212_N128_TOKENS, key
        assert hi < first or lo > last, key
        assert tuple(choice) == (64, 2), key

    def p(T, sm_count=212, out_f32=True):
        return cake_backend.k1_plan(
            arch="sm_107a", L=1, M=4096, N=128, K=T, out_f32=out_f32, sm_count=sm_count
        )

    for T in (1024, 1025, 1088, 1100, 2048, 3000, 3072, 4096, 4160, 5000, 6144, 7615):
        assert p(T) == (64, 2), T
    for T in (6656, 7168, 7680):
        assert p(T) == (64, 2), T
    for T in (512, 896, 1023, 7681, 7745, 8192, 16384):
        assert p(T) is None, T
    for T in (1024, 4096, 7680):
        assert p(T, out_f32=False) is None, T

    sms_list = (208, 210, 211, 212, 213, 216, 217)
    near = (-64, -1, 0, 1, 64)
    Ts = {T + d for T in K1_212_N128_TOKENS for d in near}
    Ts |= {8000, 8192, 12288, 16172, 16231, 16384, 65536}

    def plans():
        return {
            (sms, T, f32): p(T, sm_count=sms, out_f32=f32)
            for sms in sms_list
            for T in Ts
            for f32 in (False, True)
        }

    with_table = plans()
    monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
    without = plans()
    routed = 0
    for (sms, T, f32), plan in with_table.items():
        assert without[(sms, T, f32)] is None, (sms, T, f32)
        held = [tuple(v) for lo, hi, v in K1_212_4096_128_RANGE if lo <= T <= hi]
        if sms == 212 and f32 and held:
            assert plan == held[0], T
            routed += 1
        else:
            assert plan is None, (sms, T, f32)
    assert routed


#: The 148-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`, ``(arch, d_in, d_out, out_dtype) -> ranges``.
K1_148_EXCLUDED_TOKENS = {
    ("sm_100a", 4096, 48, "bf16"): ((512, 512),),
    ("sm_100a", 4096, 48, "f32"): ((512, 512),),
    ("sm_100a", 4096, 64, "bf16"): ((512, 512), (65536, None)),
    ("sm_100a", 4096, 64, "f32"): ((512, 512), (65536, None)),
    ("sm_100a", 5120, 48, "bf16"): ((512, 2048),),
    ("sm_100a", 5120, 48, "f32"): ((512, 512),),
    ("sm_100a", 5120, 64, "bf16"): ((512, 2048), (8192, 8192)),
    ("sm_100a", 5120, 64, "f32"): ((512, 1024), (8192, 8192)),
    ("sm_103a", 4096, 32, "bf16"): ((65536, None),),
    ("sm_103a", 4096, 32, "f32"): ((65536, None),),
    ("sm_103a", 5120, 48, "bf16"): ((512, 2048), (40961, 49151)),
    ("sm_103a", 5120, 48, "f32"): ((512, 2048),),
    ("sm_103a", 5120, 64, "bf16"): ((512, 1024), (47105, 49152)),
    ("sm_103a", 5120, 64, "f32"): ((512, 1024),),
    ("sm_103a", 5120, 128, "bf16"): ((512, 4096), (32768, 65536)),
    ("sm_103a", 5120, 128, "f32"): ((512, 4096), (32768, 65536)),
    ("sm_103a", 6144, 128, "bf16"): ((512, 8192),),
    ("sm_103a", 6144, 128, "f32"): ((512, 8192),),
}
#: The 152-SM ``sm_103a`` entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`: the FP32 ``5120 -> x`` token ranges
#: that ran a cluster of three CTAs, ``5120 -> 32`` FP32 at 16384 tokens and ``5120 -> 48`` BF16 at 16231 tokens run
#: the deterministic program.
K1_152_EXCLUDED_TOKENS = {
    ("sm_103a", 5120, 32, "f32"): ((8192, 12287), (16384, 16384)),
    ("sm_103a", 5120, 48, "bf16"): ((16231, 16231),),
    ("sm_103a", 5120, 48, "f32"): ((4096, 8192),),
    ("sm_103a", 5120, 64, "f32"): ((4096, 6143), (12288, 24576)),
    ("sm_103a", 5120, 128, "f32"): ((4097, 32768),),
}
#: The measured token counts of the 148-SM audit (40961, 47105 and 49152: points of the 10-seed FP64 error grid of
#: sm_103a ``5120 -> 48`` / ``5120 -> 64`` BF16).
K1_148_MEASURED_TOKENS = (
    512,
    1024,
    2048,
    4096,
    8192,
    16231,
    32768,
    40961,
    47105,
    49152,
    65536,
)


def test_k1_exact_sm_count_exclusions(monkeypatch):
    """A row runs the deterministic program at the token counts of its entries in
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` on a part of exactly that SM count (148-SM sm_100a / sm_103a parts,
    152-SM sm_103a parts), and every part of any other SM count (147, 149, 151, 153, 160, 208, 212, 216, 220, and 148 /
    152 for the other table) plans every row and token count as without the table.  Every entry belongs to a row of
    :data:`K1_SPLIT_RULES` with a program on that part; its ranges are closed, disjoint and not adjacent, start at a
    measured token count and end at one, one token below one, or have no upper bound; one token outside a range the
    row keeps the plan it has without the table."""
    table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM
    measured = {
        148: K1_148_MEASURED_TOKENS,
        # the cluster table's token counts and the accuracy check's (16231 and 16384)
        152: tuple(sorted({*K1_DSM_MEASURED_TOKENS[("sm_103a", 152)], 16231})),
    }
    for sms_part, want in (
        (148, K1_148_EXCLUDED_TOKENS),
        (152, K1_152_EXCLUDED_TOKENS),
    ):
        got_rows = {
            (arch, k[2], k[3], k[4]): tuple(v)
            for (arch, sms), rows in table.items()
            if sms == sms_part
            for k, v in rows.items()
        }
        assert got_rows == want, sms_part
    assert set(table) <= {("sm_100a", 148), ("sm_103a", 148), ("sm_103a", 152)}

    def plans(sms_list, Ts):
        out = {}
        for (arch, _), rows in table.items():
            for key in rows:
                _, L, M, N, dt = key
                for sms in sms_list:
                    for T in Ts:
                        out[(key, sms, T)] = cake_backend.k1_plan(
                            arch=arch,
                            L=L,
                            M=M,
                            N=N,
                            K=T,
                            out_f32=dt == "f32",
                            sm_count=sms,
                        )
        return out

    every = {t for Ts in measured.values() for t in Ts}
    Ts = sorted({T + d for T in every for d in (-1, 0, 1)} | {64, 256, 131072})
    sm_counts = (147, 148, 149, 151, 152, 153, 160, 208, 212, 216, 220)
    with_table = plans(sm_counts, Ts)
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM", {})
    without = plans(sm_counts, Ts)
    for (key, sms, T), plan in with_table.items():
        spans = table.get((key[0], sms), {}).get(key)
        if spans is None:
            assert plan == without[(key, sms, T)], (key, sms, T)
            continue
        inside = any(lo <= T and (hi is None or hi >= T) for lo, hi in spans)
        assert plan == (None if inside else without[(key, sms, T)]), (key, sms, T)
    for (arch, sms), rows in table.items():
        assert sms in measured
        part_tokens = measured[sms]
        for key, spans in rows.items():
            _, L, M, N, dt = key
            assert key[0] == arch and L == 1
            assert cake_backend.K1_SPLIT_RULES.get(key) is not None, key
            floor = max(
                cake_backend.K1_MIN_SM_COUNT[arch],
                cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
            )
            assert floor <= sms, key
            spans = sorted(spans, key=lambda r: r[0])
            for (_a, b), (c, _d) in zip(spans, spans[1:], strict=False):
                assert b is not None and b < c, key
                assert any(b < T < c for T in part_tokens), key
            for lo, hi in spans:
                # a range may start one token above a measured token count that an entry of
                # K1_SPLIT_RULE_EXCLUDED_TOKENS already holds
                assert {lo, lo - 1} & set(part_tokens), key
                assert hi is None or {hi, hi + 1} & set(part_tokens), key
                assert without[(key, sms, lo)] is not None, key
                assert with_table[(key, sms, lo)] is None, key


def test_k1_148_comb_rows_run_det_from_65536():
    """The 148-SM rows ``4096 -> 64`` (``sm_100a``) and ``4096 -> 32`` (``sm_103a``), BF16 and FP32, run the
    deterministic program from 65536 tokens up, at odd (65537, 65600, 67520) and even (65601, 67584, 98304, 131072)
    numbers of 64-token blocks, the split program one token below.  On 152 SMs ``sm_100a`` ``4096 -> 64`` runs the
    deterministic program above 65536 tokens in BF16 (its 152-SM entry of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`,
    open-ended) and in FP32 at odd numbers of 64-token blocks from 65538 tokens up
    (:data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`), and keeps the split program in FP32 at 65537 tokens and at even
    numbers; ``sm_103a`` ``4096 -> 32`` runs the deterministic program there too (its 152-SM entries of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`, open-ended)."""
    for arch, N in (("sm_100a", 64), ("sm_103a", 32)):
        for f32 in (False, True):
            kw = dict(arch=arch, L=1, M=4096, N=N, out_f32=f32)
            assert cake_backend.k1_plan(K=65535, sm_count=148, **kw) == (64, 2)
            for T in (65536, 65537, 65600, 65601, 67520, 67584, 98304, 131072):
                assert cake_backend.k1_plan(K=T, sm_count=148, **kw) is None, (kw, T)
    for f32 in (False, True):
        kw = dict(arch="sm_100a", L=1, M=4096, N=64, out_f32=f32)
        assert cake_backend.k1_plan(K=65535, sm_count=152, **kw) == (64, 2), kw
        for T in (65536, 65537, 67520, 67584, 131072):
            odd = -(-T // 64) % 2 == 1
            want = (64, 2) if f32 and (not odd or T < 65538) else None
            assert cake_backend.k1_plan(K=T, sm_count=152, **kw) == want, (kw, T)
        kw = dict(arch="sm_103a", L=1, M=4096, N=32, out_f32=f32)
        for T in (65536, 65537, 67520, 67584, 131072):
            assert cake_backend.k1_plan(K=T, sm_count=152, **kw) is None, (kw, T)


#: The 216-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` at single measured token counts outside 16172 to 16384
#: tokens, ``key -> token counts``: from 216 SMs up these rows run the deterministic program there.
K1_216_SINGLE_TOKENS = {
    ("sm_107a", 1, 5120, 32, "f32"): (512,),
    ("sm_107a", 1, 5120, 48, "f32"): (512,),
    ("sm_107a", 1, 8192, 128, "bf16"): (512, 65536),
    ("sm_107a", 1, 8192, 128, "f32"): (512, 65536),
}


def test_k1_216_single_measured_token_counts(monkeypatch):
    """The 216-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` outside the window and below the FP32 budget ranges
    are exactly the single token counts of :data:`K1_216_SINGLE_TOKENS`: on 216, 217 and 280 SMs those rows run the
    deterministic program there and keep their :data:`K1_SPLIT_RULES` verdict one token either side; 208 and 212 SMs
    plan these rows and token counts as without the entries."""
    table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    got = {}
    for key, entries in table.items():
        for sms, lo, hi in entries:
            if (
                key[0] == "sm_107a"
                and sms == 216
                and hi is not None
                and not 16172 <= lo <= 16384
            ):
                assert lo == hi, (key, lo, hi)
                got.setdefault(key, []).append(lo)
    assert {k: tuple(sorted(v)) for k, v in got.items()} == K1_216_SINGLE_TOKENS

    def plan(key, T, sms):
        _, L, M, N, dt = key
        return cake_backend.k1_plan(
            arch=key[0], L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sms
        )

    below = {
        (key, T + d, sms): plan(key, T + d, sms)
        for key, Ts in K1_216_SINGLE_TOKENS.items()
        for T in Ts
        for d in (-1, 0, 1)
        for sms in (208, 212)
    }
    for key, Ts in K1_216_SINGLE_TOKENS.items():
        rule = tuple(cake_backend.K1_SPLIT_RULES[key])
        for T in Ts:
            for sms in (216, 217, 280):
                assert plan(key, T, sms) is None, (key, T, sms)
                assert plan(key, T - 1, sms) == rule, (key, T, sms)
                assert plan(key, T + 1, sms) == rule, (key, T, sms)
    single = {(key, T) for key, Ts in K1_216_SINGLE_TOKENS.items() for T in Ts}
    stripped = {
        key: tuple(e for e in entries if not (e[0] == 216 and (key, e[1]) in single))
        for key, entries in table.items()
    }
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RULE_EXCLUDED_TOKENS", stripped)
    for (key, T, sms), want in below.items():
        assert plan(key, T, sms) == want, (key, T, sms)


def test_k1_216_sm_routing_is_the_record_table(no_k1_chains):
    """On 216-SM sm_107a parts every row decides by :data:`K1_SPLIT_RULES` (the record verdicts), except the rows of
    the 216-SM table of :data:`K1_SPLIT_OVERRIDES_BY_SM` at the token counts of :data:`K1_SPLIT_RULES_BY_SM_TOKENS`,
    and except the 216-SM token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`, which run the deterministic
    program."""
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    overrides = cake_backend.K1_SPLIT_OVERRIDES_BY_SM.get(("sm_107a", 216), {})
    for key, choice in cake_backend.K1_SPLIT_RULES.items():
        arch, L, M, N, dt = key
        if arch != "sm_107a":
            continue
        for T in (8192, 16171, 16172, 16231, 16384, 16385, 32768):
            cut = any(
                sms == 216 and lo <= T and (hi is None or hi >= T)
                for sms, lo, hi in excluded.get(key, ())
            )
            if key in overrides and 16172 <= T <= 16384:
                choice_t = overrides[key]
            else:
                choice_t = choice
            got = cake_backend.k1_plan(
                arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=216
            )
            want = None if choice_t is None or cut else tuple(choice_t)
            assert got == want, (M, N, dt, T)


def test_k1_216_sm_override_holds_only_at_its_sm_count_and_token_counts(
    monkeypatch, no_k1_chains
):
    """``4096 -> 1024`` FP32 takes the 192-wide row-major program at one split on exactly 216 SMs at 16172 to 16384
    tokens (:data:`K1_SPLIT_OVERRIDES_BY_SM`), except at 16231 and 16384 tokens (its 216-SM entries of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`: the deterministic program); one token outside that window it keeps its
    :data:`K1_SPLIT_RULES` verdict (256-wide tiles at one split) up to 40960 tokens, on 217 SMs and more it keeps that
    verdict up to 960 tokens and runs the deterministic program from 961 tokens up (its 217-SM entry of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`: the route before the 216-SM measurement), on 215 SMs (no table) it runs the
    deterministic program, and on 212 SMs its own tables decide (the 192-wide row-major program at one split in
    the window and at 512 tokens, the two-split program of :data:`K1_TWO_SPLIT_RANGES_BY_SM` inside its ranges, else the
    deterministic program).  On 216 SMs in the window the launch is the row-major-store program's: 88 tiles, 176 CTAs. A
    table entry acts on exactly its SM count, at the window's token counts only: ``None`` keeps a row on the
    deterministic program there, a verdict replaces the row's :data:`K1_SPLIT_RULES` verdict there.  These are the
    tables under :data:`K1_CHAIN_RANGES_BY_SM` (taken out here)."""
    arch, d_in, d_out = "sm_107a", 4096, 1024
    key = (arch, 1, d_in, d_out, "f32")
    overrides = cake_backend.K1_SPLIT_OVERRIDES_BY_SM
    assert overrides == {("sm_107a", 216): {key: (192, 1)}}
    assert cake_backend.K1_SPLIT_RULES[key] == (256, 1)

    def plan(sms, T, M=d_in, N=d_out):
        return cake_backend.k1_plan(
            arch=arch, L=1, M=M, N=N, K=T, out_f32=True, sm_count=sms
        )

    for T in (16172, 16200, 16300, 16383):
        assert plan(216, T) == (192, 1), T
        assert plan(217, T) is None, T
        assert plan(212, T) == (192, 1), T
        assert plan(215, T) is None, T
    # 16231 tokens: the deterministic program from 216 SMs up and on 208 / 212 SMs
    for sms in (208, 212, 215, 216, 217, 280):
        assert plan(sms, 16231) is None, sms
    # 16384 tokens: the deterministic program from 216 SMs up (a strict re-measurement on a 216-SM part) and on 212
    # SMs (the one-split program missed the accuracy check there)
    for sms in (212, 215, 216, 217, 280):
        assert plan(sms, 16384) is None, sms
    two_212 = K1_212_TWO_SPLIT_RANGES[(d_in, d_out, "f32")]
    for T in (1024, 8192, 16171, 16385, 32768, 40960):
        assert plan(216, T) == (256, 1), T
        assert plan(217, T) is None, T
        want = [v for lo, hi, v in two_212 if lo <= T <= hi]
        assert plan(212, T) == (want[0] if want else None), T
    # 217 SMs and more: the K1_SPLIT_RULES verdict up to 960 tokens, the deterministic program from 961 tokens up
    for sms in (217, 280):
        assert plan(sms, 960) == (256, 1), sms
        assert plan(sms, 961) is None, sms
    # from 40961 tokens up the deterministic program from 216 SMs up (the next tests)
    for sms in (216, 217, 280):
        for T in (40961, 65536):
            assert plan(sms, T) is None, (sms, T)
    assert plan(212, 512) == (192, 1)
    # every other row and token count on 216 SMs: the record verdict; BF16 4096 -> 1024 has none
    assert (
        cake_backend.k1_plan(
            arch=arch, L=1, M=d_in, N=d_out, K=16384, out_f32=False, sm_count=216
        )
        is None
    )
    # the planned launch on 216 SMs in the window: the 192-wide row-major-store program
    T = 16383
    G = torch.zeros(T, d_out, dtype=torch.bfloat16)
    X = torch.zeros(T, d_in, dtype=torch.bfloat16)
    out = torch.zeros(d_out, d_in, dtype=torch.float32)
    _register_k1(monkeypatch, arch)
    plan_, *_, reason = cake_backend.plan_atomic_gemm(
        X.t(), G, out, transposed_out=True, arch=arch, sm_count=216
    )
    assert (
        isinstance(plan_, cake_backend.CakeDenseGemmNondetProjectionPlan)
        and reason == ""
    )
    assert plan_.template == "dense_proj_gemm_atomic_nn_n192_f32" and not plan_.out_t
    assert (plan_.M, plan_.N, plan_.block_n, plan_.splits, plan_.tiles) == (
        d_out,
        d_in,
        192,
        1,
        88,
    )
    assert plan_.grid == (176, 1, 1) and plan_.sm_pairs == 108
    # one token below the window on 216 SMs: the 256-wide transposed-store program
    outside = cake_backend.plan_atomic_gemm(
        X[:16171].t(), G[:16171], out, transposed_out=True, arch=arch, sm_count=216
    )[0]
    assert (outside.template, outside.block_n, outside.out_t) == (
        "dense_proj_gemm_atomic_nn_n256_f32_t",
        256,
        True,
    )
    # on 212 SMs at 512 tokens: the 192-wide row-major-store program of the part's own table
    plan_ = cake_backend.plan_atomic_gemm(
        X[:512].t(), G[:512], out, transposed_out=True, arch=arch, sm_count=212
    )[0]
    assert plan_.template == "dense_proj_gemm_atomic_nn_n192_f32" and not plan_.out_t
    assert plan_.grid == (176, 1, 1) and plan_.sm_pairs == 106
    # a table entry of None keeps a row on the deterministic program in the window only; a verdict replaces the row's
    # verdict there; a table of another SM count does not act on 216 SMs (4096 -> 32 FP32: two splits on 216 SMs)
    key32 = (arch, 1, 4096, 32, "f32")
    assert cake_backend.K1_SPLIT_RULES[key32] == (64, 2)
    monkeypatch.setattr(
        cake_backend,
        "K1_SPLIT_OVERRIDES_BY_SM",
        {(arch, 216): {key32: None}, (arch, 220): {key32: (128, 2)}},
    )
    assert plan(216, 16384, N=32) is None and plan(216, 16385, N=32) == (64, 2)
    assert plan(220, 16384, N=32) == (128, 2) and plan(221, 16384, N=32) == (64, 2)


def test_k1_sm107a_token_counts_of_the_deterministic_rules(no_k1_chains):
    """The sm_107a entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` at 16172 to 16384 tokens are the single token counts
    16231 and 16384 per SM count: an entry of 208 or 212 SMs holds on exactly that SM count (not on the other table's
    part, nor from 216 SMs up), an entry of 216 SMs from 216 SMs up; the neighbouring token counts keep the row's
    verdict.  (The 216-SM ranges of the four wide FP32 rows at large token counts: the next test.)  These are the tables
    under :data:`K1_CHAIN_RANGES_BY_SM` (taken out here), whose ranges run three accumulation chains on 212 and 216 SMs.
    """

    def p(M, N, sms, T, out_f32=True):
        return cake_backend.k1_plan(
            arch="sm_107a", L=1, M=M, N=N, K=T, out_f32=out_f32, sm_count=sms
        )

    # 4096 -> 512: the deterministic program at 16231 and 16384 tokens on 208, 212 and 216 SMs
    for sms in (208, 212, 216):
        assert p(4096, 512, sms, 16231) is None and p(4096, 512, sms, 16384) is None
        for T in (16172, 16230, 16232, 16383):
            assert p(4096, 512, sms, T) == (128, 1), (sms, T)
    # 4096 -> 1024: at 16231 and 16384 tokens on 208, 212 and 216 SMs (strict re-measurements of the restored window;
    # 212 SMs at 16384: the one-split program missed the accuracy check); 212 SMs (their table) keep the 192-wide
    # row-major program below it, 208 SMs the 256-wide tiles and 216 SMs the 216-SM override
    for sms in (208, 212, 216):
        assert p(4096, 1024, sms, 16231) is None, sms
        assert p(4096, 1024, sms, 16384) is None, sms
    assert p(4096, 1024, 208, 16383) == (256, 1)
    assert p(4096, 1024, 216, 16383) == (192, 1)
    assert p(4096, 1024, 212, 16383) == (192, 1)
    # 4096 -> 48 and 4096 -> 64: the deterministic program on every SM count (their three-split verdicts are gone)
    for N in (48, 64):
        for sms in (208, 212, 216):
            assert p(4096, N, sms, 16231) is None and p(4096, N, sms, 16384) is None
    # 5120 -> 48 and 5120 -> 64: 16231 tokens on 208, 212 and 216 SMs; 16384 on 216 SMs, and for 5120 -> 48 on 208
    for N in (48, 64):
        for sms in (208, 212, 216):
            assert p(5120, N, sms, 16231) is None, (N, sms)
        assert p(5120, N, 216, 16384) is None, N
    # 212 SMs at 16384: 5120 -> 48 two splits, 5120 -> 64 the deterministic program (its verdict)
    assert p(5120, 48, 212, 16384) == (64, 2)
    assert p(5120, 64, 212, 16384) is None
    assert p(5120, 48, 208, 16384) is None
    assert p(5120, 64, 208, 16384) is None
    # 5120 -> 512: 212 SMs only
    assert p(5120, 512, 212, 16231) is None and p(5120, 512, 212, 16384) is None
    for sms in (208, 216):
        assert p(5120, 512, sms, 16231) == (128, 1), sms
    # 5120 -> 1024 FP32: no entry at these token counts; every part keeps the 256-wide one-split program at both
    for sms in (208, 212, 216):
        for T in (16231, 16384):
            assert p(5120, 1024, sms, T) == (256, 1), (sms, T)
    # 5120 -> 1024 BF16: 16231 tokens from 216 SMs up (the neighbouring token counts and 16384 keep the 256-wide
    # one-split program); 208 and 212 SMs run the deterministic program at every token count (their tables)
    for sms in (216, 217, 280):
        assert p(5120, 1024, sms, 16231, out_f32=False) is None, sms
        for T in (16172, 16230, 16232, 16384):
            assert p(5120, 1024, sms, T, out_f32=False) == (256, 1), (sms, T)
    for sms in (208, 212):
        for T in (16231, 16384):
            assert p(5120, 1024, sms, T, out_f32=False) is None, (sms, T)
    # 5120 -> 32 keeps its verdict (two splits) at both token counts; 4096 -> 32 too, except at 16231 tokens on 216 SMs
    for sms in (208, 212, 216):
        for T in (16231, 16384):
            assert p(5120, 32, sms, T) == (64, 2), (sms, T)
            if (sms, T) != (216, 16231):
                assert p(4096, 32, sms, T) == (64, 2), (sms, T)
    assert p(4096, 32, 216, 16231) is None
    # 4096 -> 4096: the deterministic program on 208, 212 and 216 SMs
    for sms in (208, 212, 216):
        assert p(4096, 4096, sms, 16172) is None


#: The first token count at which the sm_107a FP32 rows ``4096 / 5120 -> 512 / 1024`` run the deterministic program on
#: parts of 216 SMs or more: their one-split program met the FP32 accuracy budget (relative Frobenius error at most
#: 5e-5 and largest absolute error at most 1e-3 of the largest reference magnitude, against FP64) on every seed at
#: 40960 tokens and missed it from 45056 tokens up.
K1_SM107A_FP32_BUDGET_TOKENS = 40961
#: The first token count of each row's open-ended entry: ``K1_SM107A_FP32_BUDGET_TOKENS``, except ``5120 -> 1024`` from
#: 32769 (a strict re-measurement on a 216-SM part: the checkpointed step more than 1.3 % slower at 36864 and 40960).
K1_SM107A_WIDE_FP32_OPEN = {
    (4096, 512): K1_SM107A_FP32_BUDGET_TOKENS,
    (4096, 1024): K1_SM107A_FP32_BUDGET_TOKENS,
    (5120, 512): K1_SM107A_FP32_BUDGET_TOKENS,
    (5120, 1024): 32769,
}
#: The fewest SMs of the parts on which those rows run the deterministic program from 961 tokens up: the one-split
#: program's time at 961 to 40960 tokens was measured on 216-SM parts only (their 217-SM entries).
K1_SM107A_WIDE_FP32_ABOVE_216 = (217, 961)
#: Those rows ``(d_in, d_out)`` and their one-split programs ``(block_n, splits)`` below that token count.
K1_SM107A_WIDE_FP32_ROWS = {
    (4096, 512): (128, 1),
    (4096, 1024): (256, 1),
    (5120, 512): (128, 1),
    (5120, 1024): (256, 1),
}


def test_k1_sm107a_wide_fp32_rows_at_large_token_counts(no_k1_chains):
    """The one-split programs of the sm_107a FP32 rows ``4096 -> 512``, ``4096 -> 1024``, ``5120 -> 512`` and
    ``5120 -> 1024`` miss the FP32 accuracy budget at large token counts: from ``K1_SM107A_FP32_BUDGET_TOKENS`` up, with
    no upper bound, these rows run the deterministic program on parts of 216 SMs or more (their entries in
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`; ``5120 -> 1024`` from 32769 tokens, ``K1_SM107A_WIDE_FP32_OPEN``) and keep
    their one-split program below on 216 SMs (961 tokens and up included); on parts of more than 216 SMs they keep it up
    to 960 tokens only and run the deterministic program from 961 tokens up (``K1_SM107A_WIDE_FP32_ABOVE_216``: no part
    of more SMs was measured).  The 208-SM and 212-SM tables hold only at 16172 to 16384 tokens and at 512 tokens on 212
    SMs (:data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`), so those parts run the deterministic program at these token counts,
    except the 212-SM two-split ranges at either number of 64-token K blocks (``K1_212_TWO_SPLIT_RANGES``); the other
    entries of these rows lie in that window.  No other sm_107a row has a range without an upper bound, and sm_100a /
    sm_103a have no verdict for these rows.  These are the tables under :data:`K1_CHAIN_RANGES_BY_SM` (taken out here),
    whose ranges run three accumulation chains on 212 and 216 SMs."""
    rows = K1_SM107A_WIDE_FP32_ROWS
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    by_sm = cake_backend.K1_SPLIT_RULES_BY_SM
    open_ranges = {
        key
        for key, entries in excluded.items()
        if key[0] == "sm_107a" and any(hi is None for _, _, hi in entries)
    }
    assert open_ranges == {("sm_107a", 1, M, N, "f32") for M, N in rows}

    def p(M, N, sms, T, arch="sm_107a"):
        return cake_backend.k1_plan(
            arch=arch, L=1, M=M, N=N, K=T, out_f32=True, sm_count=sms
        )

    for (M, N), choice in rows.items():
        key = ("sm_107a", 1, M, N, "f32")
        T0 = K1_SM107A_WIDE_FP32_OPEN[(M, N)]
        assert cake_backend.K1_SPLIT_RULES[key] == choice
        above, first = K1_SM107A_WIDE_FP32_ABOVE_216
        assert [e for e in excluded[key] if e[2] is None] == [
            (216, T0, None),
            (above, first, None),
        ]
        for _, lo, hi in excluded[key]:
            assert hi is None or 16172 <= lo <= hi <= 16384, key
        for T in (512, 704, 961, 1024, 8192, 32768, T0 - 1):
            assert p(M, N, 216, T) == choice, (M, N, T)
        for sms in (216, 217, 280):
            for T in (T0, T0 + 4095, 45056, 65536, 65537, 131072):
                assert p(M, N, sms, T) is None, (M, N, sms, T)
        for sms in (above, 280):
            for T in (512, 704, first - 1):
                assert p(M, N, sms, T) == choice, (M, N, sms, T)
            for T in (first, 1024, 8192, 16231, 32768, T0 - 1):
                assert p(M, N, sms, T) is None, (M, N, sms, T)
        assert p(M, N, 212, 512) == tuple(by_sm[("sm_107a", 212)][key]), (M, N)
        two = K1_212_TWO_SPLIT_RANGES[(M, N, "f32")]
        for sms in (208, 212, 215):
            for T in (T0 - 1, T0, 1024, 1025, 32768, 65536, 131072):
                held = [tuple(v) for lo, hi, v in two if sms == 212 and lo <= T <= hi]
                want = held[0] if held else None
                assert p(M, N, sms, T) == want, (M, N, sms, T)
        for arch, sms in (("sm_100a", 148), ("sm_100a", 152), ("sm_103a", 152)):
            assert (arch, 1, M, N, "f32") not in cake_backend.K1_SPLIT_RULES
            for T in (T0 - 1, T0, 65536):
                assert p(M, N, sms, T, arch=arch) is None, (arch, M, N, T)


#: The entries of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`, ``key -> ((sm_count, min_tokens, max_tokens), ...)``.
K1_ODD_BLOCK_EXCLUDED_TOKENS = {
    ("sm_100a", 1, 4096, 32, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 4096, 48, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 4096, 64, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 5120, 32, "f32"): ((148, 65538, None),),
    ("sm_100a", 1, 5120, 48, "f32"): ((148, 65538, None),),
    ("sm_103a", 1, 4096, 32, "f32"): ((148, 65538, None),),
    ("sm_103a", 1, 4096, 48, "f32"): ((152, 14338, 32767), (152, 65538, None)),
    ("sm_107a", 1, 8192, 128, "f32"): ((217, 1025, 8192),),
}
#: The ranges of ``K1_ODD_BLOCK_EXCLUDED_TOKENS`` that come from the time (the backward pass or the checkpointed
#: step more than 1.3 % slower at an odd block count), not from the FP32 budget.
K1_ODD_BLOCK_TIME_RANGES = {("sm_103a", 1, 4096, 48, "f32"): ((152, 14338, 32767),)}
#: The ranges of ``K1_ODD_BLOCK_EXCLUDED_TOKENS`` that keep parts of more SMs than the measured ones on the
#: deterministic program (sm_107a ``8192 -> 128`` FP32: its time at odd block counts was measured on 212-SM and 216-SM
#: parts only).
K1_ODD_BLOCK_ABOVE_RANGES = {("sm_107a", 1, 8192, 128, "f32"): ((217, 1025, 8192),)}


def test_k1_odd_block_exclusions(monkeypatch):
    """A row of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` runs the deterministic program at a token count of an entry's
    range that spans an odd number of 64-token K blocks, on a part of the entry's SM count or more where the entry is at
    or above the row's SM-count floor and on exactly that SM count below it; at an even number and below the range the
    row keeps the route it takes without the table.  Every entry names an FP32 two-split row with a verdict on its part
    and starts at 65538 tokens (above the largest measured token count at which the two-split program met the FP32
    accuracy budget, 65537) with no upper bound, except the ranges of ``K1_ODD_BLOCK_TIME_RANGES`` (sm_103a
    ``4096 -> 48`` from 14338 to 32767 tokens on 152 SMs: the time) and of ``K1_ODD_BLOCK_ABOVE_RANGES`` (sm_107a
    ``8192 -> 128`` from 1025 to 8192 tokens from 217 SMs up: not measured there; 216 SMs keep the two-split program);
    BF16 rows have no entry.  Below 65538 tokens and outside those ranges these rows run their two-split program at
    either number of K blocks."""
    table = cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS
    assert table == K1_ODD_BLOCK_EXCLUDED_TOKENS

    def p(key, sms, T):
        arch, L, M, N, dt = key
        return cake_backend.k1_plan(
            arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sms
        )

    cases = []
    for key, entries in table.items():
        arch, L, M, N, dt = key
        assert dt == "f32" and L == 1, key
        assert (arch, L, M, N, "bf16") not in table
        floor = max(
            cake_backend.K1_MIN_SM_COUNT[arch],
            cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
        )
        for sms, lo, hi in entries:
            if (sms, lo, hi) in K1_ODD_BLOCK_TIME_RANGES.get(key, ()):
                assert sms >= floor and hi is not None and _k1_blocks(lo) % 2 == 1, key
                odd = (lo, 64 * _k1_blocks(lo), 16173, 20481, 28673, hi - 64)
                even = (64 * _k1_blocks(lo) + 1, 20480, hi)
                outside = (lo - 1, 1025)  # odd block counts below the range
                assert all(_k1_blocks(T) % 2 == 1 for T in odd + outside), key
                assert all(_k1_blocks(T) % 2 == 0 for T in even), key
                for part in (sms, sms + 1, sms + 64):
                    cases += [(key, part, T, True) for T in odd]
                    cases += [(key, part, T, False) for T in even + outside]
                continue
            if (sms, lo, hi) in K1_ODD_BLOCK_ABOVE_RANGES.get(key, ()):
                assert sms > floor and hi is not None and _k1_blocks(lo) % 2 == 1, key
                odd = (lo, 1088, 2049, 4097, hi - 64)
                even = (lo + 64, 4096, hi)
                above = (hi + 1, hi + 129)  # odd block counts above the range
                assert all(_k1_blocks(T) % 2 == 1 for T in odd + above), key
                assert all(_k1_blocks(T) % 2 == 0 for T in even), key
                for part in (sms, sms + 1, sms + 64):
                    cases += [(key, part, T, True) for T in odd]
                    cases += [(key, part, T, False) for T in even + above]
                # the measured part below the entry keeps the two-split program at these odd token counts
                cases += [(key, sms - 1, T, False) for T in odd]
                continue
            assert lo == 65538 and _k1_blocks(lo) % 2 == 1 and hi is None, key
            if sms >= floor:
                rule = cake_backend.K1_SPLIT_RULES[key]
                parts = (sms, sms + 1, sms + 64)
            else:
                rule = cake_backend.K1_SPLIT_RULES_BY_SM[(arch, sms)][key]
                parts = (sms,)
            assert rule is not None and rule[1] == 2, (key, sms)
            last = 64 * _k1_blocks(lo)  # the last token count of lo's block
            odd = (lo, last, last + 65, lo + 3072, 98305, 131073)
            even = (last + 1, 65664, 98304, 131072)
            outside = (lo - 128, lo - 1)  # odd block counts below the range
            assert all(_k1_blocks(T) % 2 == 1 for T in odd + outside), key
            assert all(_k1_blocks(T) % 2 == 0 for T in even), key
            for part in parts:
                cases += [(key, part, T, True) for T in odd]
                cases += [(key, part, T, False) for T in even + outside]
    got = {(key, part, T): p(key, part, T) for key, part, T, _ in cases}
    monkeypatch.setattr(cake_backend, "K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS", {})
    for key, part, T, inside in cases:
        want = p(key, part, T)
        if inside:
            assert got[(key, part, T)] is None, (key, part, T)
        else:
            assert got[(key, part, T)] == want, (key, part, T)
    # the entries change the route of odd token counts from 65538 up: without the table these take the two-split
    # verdict
    for key, part, T in (
        (("sm_100a", 1, 4096, 32, "f32"), 148, 65538),
        (("sm_100a", 1, 5120, 48, "f32"), 148, 98305),
        (("sm_103a", 1, 4096, 32, "f32"), 149, 65665),
    ):
        assert got[(key, part, T)] is None, (key, part, T)
        assert p(key, part, T) == (64, 2), (key, part, T)
    # below 65538 tokens odd block counts take the two-split program with the table in place (1025, 4097, 14337, 16173
    # and 65537 tokens; sm_103a 4096 -> 48 on 152 SMs not at 14338 to 32767); sm_107a ``8192 -> 128`` FP32 on 216 SMs
    # (its entry holds from 217 SMs up)
    monkeypatch.undo()
    table = cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS
    assert table == K1_ODD_BLOCK_EXCLUDED_TOKENS
    for key, part, T, want in (
        (("sm_100a", 1, 4096, 32, "f32"), 148, 1025, (64, 2)),
        (("sm_100a", 1, 4096, 64, "f32"), 152, 65537, (64, 2)),
        (("sm_100a", 1, 5120, 32, "f32"), 148, 16173, (64, 2)),
        (("sm_103a", 1, 4096, 32, "f32"), 148, 4097, (64, 2)),
        (("sm_103a", 1, 4096, 48, "f32"), 152, 14337, (64, 2)),
        (("sm_103a", 1, 4096, 48, "f32"), 152, 16173, None),
        (("sm_107a", 1, 8192, 128, "f32"), 212, 4097, (128, 2)),
        (("sm_107a", 1, 8192, 128, "f32"), 216, 1025, (128, 2)),
        (("sm_107a", 1, 8192, 128, "f32"), 216, 8127, (128, 2)),
        (("sm_107a", 1, 8192, 128, "f32"), 217, 1025, None),
        (("sm_107a", 1, 8192, 128, "f32"), 217, 1024, (128, 2)),
    ):
        assert p(key, part, T) == want, (key, part, T)


#: The ranges of :data:`K1_CHAIN_RANGES_BY_SM` (sm_107a FP32 rows), ``sm_count -> {(d_in, d_out): ((min_tokens,
#: max_tokens, (block_n, chains)), ...)}``: the one-split program of three accumulation chains at the measured token
#: counts of ``K1_CHAIN_MEASURED_TOKENS`` at which it passed the K1 rule and the bound on the backward pass and the
#: checkpointed step in every run, joined up to a failing or unmeasured one.
K1_CHAIN_RANGES = {
    212: {
        (4096, 512): ((960, 16172, (128, 3)), (32768, 40961, (128, 3))),
        (4096, 1024): ((960, 16172, (192, 3)), (32768, 40961, (192, 3))),
        (5120, 512): ((960, 16172, (128, 3)), (32768, 40961, (128, 3))),
        (5120, 1024): ((960, 40961, (256, 3)),),
    },
    216: {
        (4096, 512): ((961, 16172, (128, 3)), (32768, 40961, (128, 3))),
        (4096, 1024): ((961, 16172, (192, 3)), (32768, 40961, (192, 3))),
        (5120, 512): ((961, 40961, (128, 3)),),
        (5120, 1024): ((961, 16384, (256, 3)), (40961, 40961, (256, 3))),
    },
}
#: The token counts those rows were timed at, per SM count (212 SMs: two jobs).
K1_CHAIN_MEASURED_TOKENS = {
    212: (960, 1024, 2048, 4096, 4097, 8192, 16172, 16384, 32768, 32769, 40960, 40961),
    216: (960, 961, 1024, 2048, 4096, 8192, 16172, 16231, 16384, 32768, 40961),
}
#: The measured token counts at which the three-chain program failed: not faster than the deterministic plan in every
#: run, or (``4096 -> 1024`` at 16384 tokens on 212 SMs, ``5120 -> 1024`` at 32768 tokens on 216 SMs) the checkpointed
#: step more than 1.3 % slower.
K1_CHAIN_MISSES = {
    212: {(4096, 512): (16384,), (4096, 1024): (16384,), (5120, 512): (16384,)},
    216: {
        (4096, 512): (16231, 16384),
        (4096, 1024): (16231, 16384),
        (5120, 1024): (32768,),
    },
}


def test_k1_chain_ranges(monkeypatch):
    """On sm_107a parts of exactly 212 and 216 SMs the FP32 rows ``4096 / 5120 -> 512 / 1024`` run this package's
    one-split program of three accumulation chains at the token counts of their ranges in :data:`K1_CHAIN_RANGES_BY_SM`
    (both ends, one token inside them and the measured token counts inside): :func:`k1_chain_plan` gives the range's
    ``(block_n, chains)`` and :func:`k1_plan` that width at one split, inside the row's 212-SM two-split range of
    :data:`K1_TWO_SPLIT_RANGES_BY_SM` too; past the chain ranges the two-split ranges keep two splits at either number
    of 64-token K blocks.  Every range starts and ends at a measured token count and holds no failing one; one token
    outside a range, at a failing token count, on another SM count, on another architecture, for a BF16 output and for a
    batched launch the rows keep the routing of the other tables."""
    table = cake_backend.K1_CHAIN_RANGES_BY_SM
    assert cake_backend.K1_CHAINS_CHOICES == (1, 3)
    assert set(table) == {("sm_107a", sms) for sms in K1_CHAIN_RANGES}
    failing = {}
    for sms, rows in K1_CHAIN_RANGES.items():
        part = ("sm_107a", sms)
        assert {(k[2], k[3]) for k in table[part]} == set(rows)
        measured = K1_CHAIN_MEASURED_TOKENS[sms]
        for (d_in, d_out), spans in rows.items():
            key = ("sm_107a", 1, d_in, d_out, "f32")
            assert tuple(table[part][key]) == spans, key
            kw = dict(arch="sm_107a", L=1, M=d_in, N=d_out, out_f32=True, sm_count=sms)
            misses = K1_CHAIN_MISSES[sms].get((d_in, d_out), ())
            for (_a, b, _v), (c, _d, _w) in zip(spans, spans[1:], strict=False):
                assert b < c and any(b < T < c for T in misses), key
            for lo, hi, (block_n, chains) in spans:
                assert lo in measured and hi in measured, key
                assert not any(lo <= T <= hi for T in misses), key
                assert chains in cake_backend.K1_CHAINS_CHOICES and chains > 1, key
                template = cake_backend.atomic_k1_template(block_n, True, chains)
                assert template in cake_jit.KERNELS["sm_107a"], template
                inside = {lo, hi, min(lo + 1, hi)} | {
                    T for T in measured if lo <= T <= hi
                }
                for T in sorted(inside):
                    got = cake_backend.k1_chain_plan(K=T, **kw)
                    assert got == (block_n, chains), (key, sms, T)
                    assert cake_backend.k1_plan(K=T, **kw) == (block_n, 1), (key, T)
                for T in (lo - 1, hi + 1):
                    if not any(a <= T <= b for a, b, _ in spans):
                        assert cake_backend.k1_chain_plan(K=T, **kw) is None, (key, T)
            for T in misses:
                assert cake_backend.k1_chain_plan(K=T, **kw) is None, (key, sms, T)
                failing[(key, sms, T)] = cake_backend.k1_plan(K=T, **kw)
            for T in (961, 4097, 16172, 40961):
                for other in (208, 210, 211, 213, 215, 217, 220):
                    assert (
                        cake_backend.k1_chain_plan(K=T, **dict(kw, sm_count=other))
                        is None
                    ), (key, other, T)
                assert (
                    cake_backend.k1_chain_plan(K=T, **dict(kw, out_f32=False)) is None
                )
                assert cake_backend.k1_chain_plan(K=T, **dict(kw, L=2)) is None
                for arch in ("sm_100a", "sm_103a"):
                    assert (
                        cake_backend.k1_chain_plan(K=T, **dict(kw, arch=arch)) is None
                    ), (key, arch, T)
    # on 212 SMs the token counts of the two-split ranges inside the chain ranges run three chains: 4096 -> 512 at
    # 1024 and 8192 tokens, 4096 -> 1024 at 2048 tokens, 5120 -> 1024 at 1024 tokens, the 512-wide rows at 32768
    # tokens; past the chain ranges the two-split ranges keep two splits (40962 and 45057 tokens: 641 and 705 blocks;
    # 65535 and 65536 tokens: 1024 blocks)
    p212 = dict(arch="sm_107a", L=1, out_f32=True, sm_count=212)
    for M, N, T, block_n in (
        (4096, 512, 1024, 128),
        (4096, 512, 8192, 128),
        (4096, 1024, 2048, 192),
        (5120, 1024, 1024, 256),
        (4096, 512, 32768, 128),
        (5120, 512, 32768, 128),
    ):
        assert cake_backend.k1_chain_plan(M=M, N=N, K=T, **p212) == (block_n, 3)
        assert cake_backend.k1_plan(M=M, N=N, K=T, **p212) == (block_n, 1)
    for M, N, block_n in ((4096, 512, 128), (4096, 1024, 192), (5120, 512, 128)):
        for T in (40962, 45057, 65535, 65536):
            assert cake_backend.k1_chain_plan(M=M, N=N, K=T, **p212) is None
            assert cake_backend.k1_plan(M=M, N=N, K=T, **p212) == (block_n, 2), T
    # 5120 -> 1024 has no two-split range past 8192 tokens on 212 SMs: three chains at 32768 tokens
    assert cake_backend.k1_chain_plan(M=5120, N=1024, K=32768, **p212) == (256, 3)
    # on 216 SMs up to 960 tokens the one-chain program of K1_SPLIT_RULES; past the chain ranges, from 40962 tokens
    # up, the deterministic program
    for (d_in, d_out), choice in K1_SM107A_WIDE_FP32_ROWS.items():
        kw = dict(arch="sm_107a", L=1, M=d_in, N=d_out, out_f32=True, sm_count=216)
        assert cake_backend.k1_chain_plan(K=960, **kw) is None
        assert cake_backend.k1_plan(K=960, **kw) == choice
        for T in (40962, 65536):
            assert cake_backend.k1_chain_plan(K=T, **kw) is None
            assert cake_backend.k1_plan(K=T, **kw) is None, (d_in, d_out, T)
    # at a failing token count the rows keep the routing of the other tables: the plan without the chain ranges
    # (216 SMs: 4096 -> 1024 at 16384 tokens the deterministic program, 5120 -> 1024 at 32768 tokens one chain)
    assert failing
    monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})
    for (key, sms, T), plan in failing.items():
        _, L, M, N, _ = key
        want = cake_backend.k1_plan(
            arch="sm_107a", L=L, M=M, N=N, K=T, out_f32=True, sm_count=sms
        )
        assert plan == want, (key, sms, T)
    p216 = dict(arch="sm_107a", L=1, out_f32=True, sm_count=216)
    assert failing[(("sm_107a", 1, 4096, 1024, "f32"), 216, 16384)] is None
    assert failing[(("sm_107a", 1, 5120, 1024, "f32"), 216, 32768)] == (256, 1)
    for M, N in ((4096, 512), (4096, 1024)):
        assert cake_backend.k1_plan(M=M, N=N, K=16231, **p216) is None


def test_k1_chains_fit(monkeypatch):
    """``chains * 64 <= T + 63``: every chain gets at least one 64-token K block; a token count of a chain range too
    short for its chains keeps the routing of the other tables."""
    assert cake_backend.k1_chains_fit(3, 129)
    assert not cake_backend.k1_chains_fit(3, 128)
    assert cake_backend.k1_chains_fit(1, 1)
    assert not cake_backend.k1_chains_fit(0, 4096)
    key = ("sm_107a", 1, 4096, 512, "f32")
    tables = {("sm_107a", 212): {key: ((64, 256, (128, 3)),)}}
    monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", tables)
    monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
    kw = dict(arch="sm_107a", L=1, M=4096, N=512, out_f32=True, sm_count=212)
    assert cake_backend.k1_chain_plan(K=129, **kw) == (128, 3)
    assert cake_backend.k1_chain_plan(K=128, **kw) is None


def test_k1_chain_plans_run_the_chain_programs(monkeypatch):
    """A token count of :data:`K1_CHAIN_RANGES_BY_SM` plans the one-split program of three chains of the range's width
    (``CakeDenseGemmNondetProjectionPlan.chains``, template ``..._c3``, no workspace and no tickets): the 128-wide and 256-wide
    transposed-store programs and the 192-wide row-major-store program, inside a 212-SM two-split range too; past the
    chain range the two-split range plans two splits of one chain.  An explicit ``splits`` / ``block_n`` runs one chain;
    ``chains=3`` names the three-chain program of any row at one split; a chain count outside :data:`K1_CHAINS_CHOICES`,
    ``chains=3`` with more than one split or on fewer than three K blocks raise ``ValueError``; a BF16 three-chain
    program is not registered.  :data:`K1_PARITY_RANGES_BY_SM`, which comes first at these token counts, is taken out
    (``test_k1_parity_ranges_plan_their_program`` holds it)."""
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    arch = "sm_107a"
    _register_k1(monkeypatch, arch)
    kw = dict(transposed_out=True, arch=arch, sm_count=212)

    def operands(T, d_in, d_out, dtype=torch.float32):
        G = torch.zeros(T, d_out, dtype=torch.bfloat16)
        X = torch.zeros(T, d_in, dtype=torch.bfloat16)
        return X.t(), G, torch.zeros(d_out, d_in, dtype=dtype)

    for d_in, d_out, template, out_t in (
        (4096, 512, "dense_proj_gemm_atomic_nn_n128_f32_t_c3", True),
        (5120, 1024, "dense_proj_gemm_atomic_nn_n256_f32_t_c3", True),
        (4096, 1024, "dense_proj_gemm_atomic_nn_n192_f32_c3", False),
    ):
        plan, *_, reason = cake_backend.plan_atomic_gemm(
            *operands(4097, d_in, d_out), **kw
        )
        assert (
            isinstance(plan, cake_backend.CakeDenseGemmNondetProjectionPlan)
            and reason == ""
        )
        assert (plan.template, plan.splits, plan.chains, plan.out_t) == (
            template,
            1,
            3,
            out_t,
        )
        assert plan.ws_f32_elems == 0 and plan.counters_u32 == 0
        assert template in cake_jit.KERNELS[arch]
        # an explicit split count or width runs one chain
        one = cake_backend.plan_atomic_gemm(
            *operands(4097, d_in, d_out), splits=1, **kw
        )[0]
        assert one.chains == 1 and not one.template.endswith("_c3")
        assert one.block_n == plan.block_n
    plan = cake_backend.plan_atomic_gemm(*operands(4096, 4096, 512), **kw)[0]
    assert (plan.block_n, plan.splits, plan.chains) == (128, 1, 3)
    plan = cake_backend.plan_atomic_gemm(*operands(65536, 4096, 512), **kw)[0]
    assert (plan.block_n, plan.splits, plan.chains) == (128, 2, 1)
    # an explicit three-chain program on a row without a chain range (sm_103a, 152 SMs)
    kw103 = dict(transposed_out=True, arch="sm_103a", sm_count=152)
    _register_k1(monkeypatch, "sm_103a")
    plan = cake_backend.plan_atomic_gemm(
        *operands(16231, 6144, 128), chains=3, **kw103
    )[0]
    assert (plan.template, plan.splits, plan.chains) == (
        "dense_proj_gemm_atomic_nn_n128_f32_t_c3",
        1,
        3,
    )
    plan = cake_backend.plan_atomic_gemm(
        *operands(16231, 6144, 128), chains=3, block_n=256, **kw103
    )[0]
    assert plan.template == "dense_proj_gemm_atomic_nn_n256_f32_t_c3"
    with pytest.raises(ValueError, match="chains is one of"):
        cake_backend.plan_atomic_gemm(*operands(16231, 6144, 128), chains=2, **kw103)
    with pytest.raises(ValueError, match="at one split"):
        cake_backend.plan_atomic_gemm(
            *operands(16231, 6144, 128), chains=3, splits=2, **kw103
        )
    with pytest.raises(ValueError, match="at one split"):
        cake_backend.plan_atomic_gemm(*operands(128, 6144, 128), chains=3, **kw103)
    plan, *_, reason = cake_backend.plan_atomic_gemm(
        *operands(16231, 6144, 128, torch.bfloat16), chains=3, **kw103
    )
    assert plan is None and "dense_proj_gemm_atomic_nn_n128_t_c3" in reason
    assert cake_backend.atomic_k1_template(192, True, 3) == (
        "dense_proj_gemm_atomic_nn_n192_f32_c3"
    )
    assert cake_backend.atomic_k1_template(128, True, 1) == (
        "dense_proj_gemm_atomic_nn_n128_f32_t"
    )


def test_k1_chain_programs_are_registered():
    """The chain programs ship on the architectures of their one-chain programs: the three-chain 128-wide and 256-wide
    FP32 transposed-store programs on sm_100a, sm_103a and sm_107a, the three-chain 192-wide FP32 row-major-store
    program on sm_107a, and for :data:`K1_ORDERED_RANGES_BY_SM` on sm_107a the two- and four-chain 192-wide row-major
    and 256-wide transposed-store FP32 programs and the two-chain 128-wide one, for :data:`K1_PARITY_RANGES_BY_SM`
    on sm_107a the four-chain 128-wide and the eight-chain 256-wide transposed-store FP32 programs, on sm_100a the
    two-chain 256-wide transposed-store BF16 program; each is the one-chain program's build with ``chains`` set
    (``split_order`` at its default, "arrival", whether the record names it or not)."""
    want = {
        "dense_proj_gemm_atomic_nn_n128_f32_t_c3": ("sm_100a", "sm_103a", "sm_107a"),
        "dense_proj_gemm_atomic_nn_n256_f32_t_c3": ("sm_100a", "sm_103a", "sm_107a"),
        "dense_proj_gemm_atomic_nn_n192_f32_c3": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n192_f32_c2": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n128_f32_t_c2": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n128_f32_t_c4": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n256_f32_t_c2": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n192_f32_c4": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n256_f32_t_c4": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n256_f32_t_c8": ("sm_107a",),
        "dense_proj_gemm_atomic_nn_n256_t_c2": ("sm_100a",),
    }
    got = {}
    for arch, kernels in cake_jit.KERNELS.items():
        for template in kernels:
            if template.endswith(("_c2", "_c3", "_c4", "_c8")):
                got.setdefault(template, []).append(arch)
    assert {t: tuple(sorted(a)) for t, a in got.items()} == want
    for template, arches in want.items():
        one, chains = template[: -len("_c3")], int(template[-1])
        for arch in arches:
            chained = cake_jit.MODULES[cake_jit.KERNELS[arch][template]]
            single = cake_jit.MODULES[cake_jit.KERNELS[arch][one]]
            assert chained["build"]["module"] == single["build"]["module"]
            # ``split_order`` defaults to "arrival" in the build; one split per tile makes the order moot
            assert {"split_order": "arrival", **chained["build"]["kwargs"]} == {
                "split_order": "arrival",
                **single["build"]["kwargs"],
                "chains": chains,
            }
            assert chained["arg_plan"] == single["arg_plan"]
            assert chained["launch"] == single["launch"]


def test_k1_plan_is_the_transposed_store_launch(monkeypatch):
    """The thin weight gradient ``out[32, 4096] = G.T @ X``: either orientation of the caller's views gives the same
    launch -- ``A = X`` ``[L, T, d_in]``, ``B = G`` ``[L, T, d_out]``, ``M = d_in``, ``N = d_out`` -- the planner's
    tile and split count, the workspace sizes, the grid; an unregistered template keeps the row on the deterministic
    program."""
    arch, T, N, K = "sm_103a", 16231, 32, 4096
    G, X = (
        torch.zeros(T, N, dtype=torch.bfloat16),
        torch.zeros(T, K, dtype=torch.bfloat16),
    )
    out = torch.zeros(N, K, dtype=torch.bfloat16)
    kw = dict(arch=arch, sm_count=152)
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    plan, *_, reason = cake_backend.plan_atomic_gemm(
        X.t(), G, out, transposed_out=True, **kw
    )
    assert plan is None and "not registered" in reason
    _register_k1(monkeypatch, arch)
    swapped = cake_backend.plan_atomic_gemm(G.t(), X, out, transposed_out=False, **kw)
    for plan, a_desc, b_desc, out3, reason in (
        cake_backend.plan_atomic_gemm(X.t(), G, out, transposed_out=True, **kw),
        swapped,
    ):
        assert (
            isinstance(plan, cake_backend.CakeDenseGemmNondetProjectionPlan)
            and reason == ""
        )
        assert (plan.L, plan.M, plan.N, plan.K, plan.block_n, plan.splits) == (
            1,
            K,
            N,
            T,
            64,
            2,
        )
        assert (plan.m_tiles, plan.n_tiles, plan.tiles) == (16, 1, 16)
        assert plan.grid == (2 * 32, 1, 1)
        assert plan.template == "dense_proj_gemm_atomic_nn_n64_t"
        assert plan.ws_f32_elems == 16 * 2 * 256 * 64
        assert plan.counters_u32 == 16 * 512
        assert tuple(a_desc.shape) == (1, T, K) and a_desc.stride(-1) == 1
        assert tuple(b_desc.shape) == (1, T, N) and b_desc.stride(-1) == 1
        assert a_desc.data_ptr() == X.data_ptr() and b_desc.data_ptr() == G.data_ptr()
        assert tuple(out3.shape) == (1, N, K)
    assert swapped[0].swapped and not swapped[0].transposed_out


def test_k1_explicit_choices(monkeypatch):
    arch, T, N, K = "sm_103a", 1000, 128, 6144
    _register_k1(monkeypatch, arch)
    G = torch.zeros(T, N, dtype=torch.bfloat16)
    X = torch.zeros(T, K, dtype=torch.bfloat16)
    out = torch.zeros(N, K, dtype=torch.float32)
    kw = dict(transposed_out=True, arch=arch, sm_count=152)
    plan = cake_backend.plan_atomic_gemm(X.t(), G, out, splits=1, **kw)[0]
    assert (plan.splits, plan.block_n) == (1, 128)  # the planner's tile width
    assert plan.ws_f32_elems == 0 and plan.counters_u32 == 0
    assert plan.template == "dense_proj_gemm_atomic_nn_n128_f32_t"
    plan = cake_backend.plan_atomic_gemm(X.t(), G, out, splits=16, block_n=256, **kw)[0]
    assert (plan.splits, plan.block_n, plan.n_tiles, plan.items) == (16, 256, 1, 384)
    assert plan.grid == (2 * 76, 1, 1)
    with pytest.raises(ValueError, match="splits must be in"):
        cake_backend.plan_atomic_gemm(X.t(), G, out, splits=17, **kw)
    with pytest.raises(ValueError, match="block_n"):
        cake_backend.plan_atomic_gemm(X.t(), G, out, block_n=96, **kw)
    # a row the planner does not take still runs an explicit choice
    G2 = torch.zeros(T, 1024, dtype=torch.bfloat16)
    out2 = torch.zeros(1024, K, dtype=torch.bfloat16)
    assert cake_backend.plan_atomic_gemm(X.t(), G2, out2, **kw)[0] is None
    plan = cake_backend.plan_atomic_gemm(X.t(), G2, out2, splits=3, **kw)[0]
    assert (plan.splits, plan.block_n) == (3, cake_backend.k1_block_n(1024))


def test_k1_planner_keeps_rows_it_cannot_serve_on_the_deterministic_program(
    monkeypatch,
):
    arch, T = "sm_103a", 1000
    _register_k1(monkeypatch, arch)
    kw = dict(transposed_out=False, arch=arch, sm_count=152, splits=2)

    def reason(A, B, out):
        plan, *_, why = cake_backend.plan_atomic_gemm(A, B, out, **kw)
        assert plan is None
        return why

    G = torch.zeros(T, 64, dtype=torch.bfloat16)
    X = torch.zeros(T, 512, dtype=torch.bfloat16)
    out = torch.zeros(64, 512, dtype=torch.bfloat16)
    assert cake_backend.plan_atomic_gemm(G.t(), X, out, **kw)[0] is not None
    assert "K-major" in reason(torch.zeros(64, T, dtype=torch.bfloat16), X, out)
    assert "bf16" in reason(G.t().float(), X, out)
    odd = torch.zeros(T, 64, dtype=torch.bfloat16)[:, :60]
    assert "M or N not a multiple of 8" in reason(
        odd.t(), X, torch.zeros(60, 512, dtype=torch.bfloat16)
    )
    unaligned = torch.zeros(T, 60, dtype=torch.bfloat16)
    assert "multiples of 8" in reason(
        unaligned.t(), X, torch.zeros(60, 512, dtype=torch.bfloat16)
    )
    shifted = torch.zeros(64 * 512 + 4, dtype=torch.bfloat16)[4:].view(64, 512)
    assert "output view" in reason(G.t(), X, shifted)
    with pytest.raises(ValueError, match="out must be"):
        cake_backend.plan_atomic_gemm(G.t(), X, out.t(), **kw)
    assert [cake_backend.k1_block_n(n) for n in (32, 48, 64, 128, 576, 1024, 4096)] == [
        64,
        64,
        64,
        128,
        64,
        256,
        256,
    ]


def test_k1_row_major_verdict_runs_the_row_major_program(monkeypatch, no_k1_chains):
    """A 192-wide verdict (:data:`K1_ROW_MAJOR_BLOCK_N`) runs the row-major-store program with the operands of the
    transposed-store launch exchanged: ``A = G`` ``[L, T, d_out]``, ``B = X`` ``[L, T, d_in]``, ``M = d_out``,
    ``N = d_in``, tiles of 256 ``d_out`` rows by 192 ``d_in`` columns, the caller's ``out[d_out, d_in]`` stored row by
    row; either orientation of the caller's views gives that launch.  An explicit ``block_n=192`` takes the same
    program; :func:`k1_block_n` never picks it."""
    arch, T, d_in, d_out = "sm_107a", 16231, 4096, 1024
    G = torch.zeros(T, d_out, dtype=torch.bfloat16)
    X = torch.zeros(T, d_in, dtype=torch.bfloat16)
    out = torch.zeros(d_out, d_in, dtype=torch.float32)
    key = (arch, 1, d_in, d_out, "f32")
    tables = {k: dict(v) for k, v in cake_backend.K1_SPLIT_RULES_BY_SM.items()}
    tables[(arch, 212)][key] = (192, 1)
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RULES_BY_SM", tables)
    monkeypatch.delitem(cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS, key, raising=False)
    _register_k1(monkeypatch, arch)
    kw = dict(arch=arch, sm_count=212)
    assert cake_backend.k1_plan(L=1, M=d_in, N=d_out, K=T, out_f32=True, **kw) == (
        192,
        1,
    )
    for A, B, transposed_out in ((X.t(), G, True), (G.t(), X, False)):
        plan, a_desc, b_desc, out3, reason = cake_backend.plan_atomic_gemm(
            A, B, out, transposed_out=transposed_out, **kw
        )
        assert (
            isinstance(plan, cake_backend.CakeDenseGemmNondetProjectionPlan)
            and reason == ""
        )
        assert plan.template == "dense_proj_gemm_atomic_nn_n192_f32"
        assert not plan.out_t
        assert (plan.L, plan.M, plan.N, plan.K, plan.block_n, plan.splits) == (
            1,
            d_out,
            d_in,
            T,
            192,
            1,
        )
        assert (plan.m_tiles, plan.n_tiles, plan.tiles) == (4, 22, 88)
        assert plan.grid == (176, 1, 1)
        assert plan.ws_f32_elems == 0 and plan.counters_u32 == 0
        assert plan.swapped == transposed_out
        assert a_desc.data_ptr() == G.data_ptr() and tuple(a_desc.shape) == (
            1,
            T,
            d_out,
        )
        assert b_desc.data_ptr() == X.data_ptr() and tuple(b_desc.shape) == (1, T, d_in)
        assert a_desc.stride(-1) == 1 and b_desc.stride(-1) == 1
        assert tuple(out3.shape) == (1, d_out, d_in) and out3.stride(1) == d_in
        views = cake_backend._k1_launch_views(A, B, out, plan)
        assert (
            views[0].data_ptr() == G.data_ptr() and views[1].data_ptr() == X.data_ptr()
        )
    # an explicit 192-wide choice is the same program, at any split count; the planner's other rows are untouched
    kwt = dict(transposed_out=True, **kw)
    plan = cake_backend.plan_atomic_gemm(X.t(), G, out, block_n=192, splits=2, **kwt)[0]
    assert (plan.template, plan.out_t, plan.splits, plan.items) == (
        "dense_proj_gemm_atomic_nn_n192_f32",
        False,
        2,
        176,
    )
    assert plan.ws_f32_elems == 176 * 256 * 192 and plan.counters_u32 == 88 * 512
    assert cake_backend.k1_block_n(192) == 64 and cake_backend.k1_block_n(384) == 128
    # a part that does not register the row-major program keeps the row on the deterministic program
    monkeypatch.setitem(
        cake_jit.KERNELS,
        arch,
        {
            t: n
            for t, n in cake_jit.KERNELS[arch].items()
            if t != "dense_proj_gemm_atomic_nn_n192_f32"
        },
    )
    plan, *_, reason = cake_backend.plan_atomic_gemm(X.t(), G, out, **kwt)
    assert plan is None and "not registered" in reason
    assert (
        cake_backend.atomic_k1_template(192, False) == "dense_proj_gemm_atomic_nn_n192"
    )


# ---------------------------------------------------------------------------
# K2 planner (CPU)
# ---------------------------------------------------------------------------


def _span_first(span, steps, spans, head_spans=0, head_steps=0):
    """The program's first step of span ``span``: the head spans share the first ``head_steps`` steps evenly, the
    others the rest (every span alike when ``head_spans`` is 0)."""
    if span < head_spans:
        return span * head_steps // head_spans
    return head_steps + (span - head_spans) * (steps - head_steps) // (
        spans - head_spans
    )


def _segments(m_pairs, n_tiles, k_iters, units, head_spans=0, head_steps=0):
    """Each tile's segments ``(k_first, k_end)`` in K blocks, in K order, tile ``pair * n_tiles + column``: span
    ``s`` runs the (row-tile pair, K block) steps ``[first(s), first(s + 1))`` (:func:`_span_first`), pair-major."""
    steps, spans = m_pairs * k_iters, units // n_tiles
    per_tile = [[] for _ in range(m_pairs * n_tiles)]
    for span in range(spans):
        first = _span_first(span, steps, spans, head_spans, head_steps)
        last = _span_first(span + 1, steps, spans, head_spans, head_steps)
        step = first
        while step < last:
            pair, k_first = divmod(step, k_iters)
            k_end = min(k_iters, k_first + last - step)
            for column in range(n_tiles):
                per_tile[pair * n_tiles + column].append((k_first, k_end))
            step += k_end - k_first
    return per_tile


@pytest.mark.parametrize(
    "m_pairs, n_tiles, k_iters",
    [(24, 2, 508), (64, 2, 192), (17, 2, 129), (1, 1, 3), (5, 3, 7)],
)
def test_router_partition_helpers_match_the_segments(m_pairs, n_tiles, k_iters):
    tiles = m_pairs * n_tiles
    for spec in ("tiles", "pairs", "forward", "S2", "S3", 6, 1000000):
        for sm_count in (148, 152, 216):
            units = cake_backend.router_units_of(
                spec, m_pairs, n_tiles, k_iters, sm_count
            )
            assert units % n_tiles == 0 and n_tiles <= units <= tiles * k_iters
            per_tile = _segments(m_pairs, n_tiles, k_iters, units)
            assert all(
                sum(e - f for f, e in segs) == k_iters for segs in per_tile
            )  # every tile covered once
            part = any(len(segs) > 1 for segs in per_tile)
            assert (
                cake_backend.router_has_part_tiles(m_pairs, n_tiles, k_iters, units)
                == part
            )
            if cake_backend.router_at_most_two_partials(m_pairs, n_tiles, units):
                assert max(len(segs) for segs in per_tile) <= 2
    for spec in (
        "H0.5x3",
        "H0.5x4",
        "H0.47x3",
        "H0.9x2",
        "H0.1x1",
        "H1x4p44",
        "H1x6p44",
        "H0.75x4p3",
    ):
        for sm_count in (148, 152, 216):
            units, head_spans, head_steps = cake_backend.router_partition_of(
                spec, m_pairs, n_tiles, k_iters, sm_count
            )
            assert units % n_tiles == 0 and 0 < head_spans < units // n_tiles
            assert (
                cake_backend.router_units_of(spec, m_pairs, n_tiles, k_iters, sm_count)
                == units
            )
            if not cake_backend.router_partition_fits(
                m_pairs, n_tiles, k_iters, units, head_spans, head_steps
            ):
                continue
            per_tile = _segments(
                m_pairs, n_tiles, k_iters, units, head_spans, head_steps
            )
            assert all(sum(e - f for f, e in segs) == k_iters for segs in per_tile)
            part = any(len(segs) > 1 for segs in per_tile)
            assert (
                cake_backend.router_has_part_tiles(
                    m_pairs, n_tiles, k_iters, units, head_spans, head_steps
                )
                == part
            )


def test_router_head_tail_partition():
    """``"H<f>x<k>"``: one head unit per CTA pair over about the fraction ``f`` of the (row-tile pair, K block)
    steps, then ``k`` tail units per CTA pair over the rest; each count a multiple of the column tile count."""
    # hidden 5120 (20 row-tile pairs x 2 column tiles), T = 16231 (508 K blocks): 10160 steps
    assert cake_backend.router_partition_of("H0.5x4", 20, 2, 508, 216) == (
        540,
        54,
        5080,
    )
    assert cake_backend.router_partition_of("H0.5x3", 20, 2, 508, 216) == (
        432,
        54,
        5080,
    )
    assert cake_backend.router_partition_of("H0.5x4", 20, 2, 508, 152) == (
        380,
        38,
        5080,
    )
    assert cake_backend.router_partition_of("H0.5x3", 20, 2, 508, 152) == (
        304,
        38,
        5080,
    )
    # hidden 4096 / 6144 at T = 16172 (506 K blocks)
    assert cake_backend.router_partition_of("H0.5x4", 16, 2, 506, 216) == (
        540,
        54,
        4048,
    )
    assert cake_backend.router_partition_of("H0.5x4", 24, 2, 506, 216) == (
        540,
        54,
        6072,
    )
    # the head share stays at least one step per head span and leaves one per tail span
    assert cake_backend.router_partition_of("H0.001x1", 20, 2, 508, 216) == (
        216,
        54,
        54,
    )
    assert cake_backend.router_partition_of("H0.999x1", 20, 2, 508, 216) == (
        216,
        54,
        10106,
    )
    assert cake_backend.router_head_tail("H0.5x4") == (0.5, 4)
    assert [
        cake_backend.router_head_tail(s) for s in ("waves", "S2", 6, "H0.5", "Hx4")
    ] == [None] * 5
    # "H<f>x<k>p<h>": h head units, each about f of a pair's even share of the steps; hidden 4096 (16 row-tile
    # pairs), T = 16172 (506 K blocks): 8096 steps, 44 of the 76 CTA pairs at 152 SMs -> 1 * 8096 * 44 / 76 = 4687
    # head steps, then 4 tail units per CTA pair
    assert cake_backend.router_partition_of("H1x4p44", 16, 2, 506, 152) == (
        348,
        22,
        4687,
    )
    # hidden 6144 (24 row-tile pairs): 12144 steps -> 7031 head steps; 6 tail units per CTA pair
    assert cake_backend.router_partition_of("H1x6p44", 24, 2, 506, 152) == (
        500,
        22,
        7031,
    )
    # one head unit per CTA pair (or more requested): the "H<f>x<k>" partition
    for h in (76, 77, 500):
        assert cake_backend.router_partition_of(
            f"H0.5x3p{h}", 20, 2, 508, 152
        ) == cake_backend.router_partition_of("H0.5x3", 20, 2, 508, 152)
    assert cake_backend.router_head_tail("H1x4p44") == (1.0, 4)
    assert cake_backend.router_head_pairs("H1x4p44") == 44
    assert [
        cake_backend.router_head_pairs(s) for s in ("H0.5x4", "waves", 6, "H1x4p")
    ] == [None] * 4
    # a token count too short for the partition: at most one unit per (tile, K block) step
    units, head_spans, head_steps = cake_backend.router_partition_of(
        "H0.5x4", 20, 2, 4, 216
    )
    assert not cake_backend.router_partition_fits(
        20, 2, 4, units, head_spans, head_steps
    )
    assert cake_backend.router_partition_fits(20, 2, 508, 540, 54, 5080)


def test_router_waves_partition():
    """``"waves"``: ``c`` whole waves of the CTA pairs, ``c = max(2, round(tiles * k_iters / (68 * pairs))) + 1``,
    only where the deterministic plan (``det_splits`` per tile, by default the deterministic package's default count,
    6) fills its last wave below 0.8.  No weight-gradient row of the deterministic package's ``ROUTER_SPLIT_RULES``
    meets a ``"waves"`` window of a router table (same architecture and hidden size, overlapping token counts): the
    windows were measured against the default count, which is that package's count at every token count they hold."""
    assert det_backend.ROUTER_SPLITS["nn_t"] == cake_backend.ROUTER_DET_WGRAD_SPLITS
    waves = [
        (key[0], key[2], lo, hi)
        for table in (
            cake_backend.ROUTER_RULES,
            *cake_backend.ROUTER_RULES_BY_SM.values(),
        )
        for key, rule in table.items()
        if key[1] == "nn_t" and isinstance(rule, tuple)
        for lo, hi, spec in rule
        if spec == "waves"
    ]
    assert waves
    for (arch, _, layout, hidden), windows in det_backend.ROUTER_SPLIT_RULES.items():
        for lo, hi, _ in windows:
            assert layout != "nn_t" or not any(
                (a, h) == (arch, hidden) and lo <= w_hi and w_lo <= hi
                for a, h, w_lo, w_hi in waves
            ), (arch, hidden, lo, hi)
    # hidden 5120 (20 row-tile pairs x 2 column tiles), T = 16231 (508 K blocks)
    assert cake_backend.router_waves(20, 2, 508, 152) == 5 * 76
    assert cake_backend.router_waves(20, 2, 508, 216) == 4 * 108
    assert cake_backend.router_waves(20, 2, 506, 152) == 5 * 76  # T = 16172
    # 148 SMs: 240 / 74 = 3.24 split waves, the last 81 % full -> deterministic
    assert cake_backend.router_waves(20, 2, 508, 148) is None
    # hidden 4096 / 6144: fuller deterministic waves (0.84 / 0.95 on 76 pairs, 0.89 / 0.89 on 108)
    for m_pairs in (16, 24):
        for sms in (152, 216):
            assert cake_backend.router_waves(m_pairs, 2, 508, sms) is None
    # few tiles: at least two waves, plus one
    assert cake_backend.router_waves(8, 2, 508, 152) == 3 * 76
    assert cake_backend.router_waves(20, 2, 129, 152) == 3 * 76
    # another deterministic split count: nine fill the last wave of 76 pairs to 95 % (deterministic), three to 79 %
    assert cake_backend.router_waves(20, 2, 508, 152, 9) is None
    assert cake_backend.router_waves(20, 2, 508, 152, 3) == 5 * 76


def test_router_partition_waves_weighs_the_deterministic_count_of_the_row(monkeypatch):
    """A ``"waves"`` window weighs the plan the row runs without it: the deterministic package's split count of the
    row (its ``router_splits``), so a row of its ``ROUTER_SPLIT_RULES`` in the window moves the partition's condition
    with it.  Hidden 5120 on a 153-SM ``sm_103a`` part (the ``"waves"`` window under the 152-SM SIMT window): six
    splits fill the last wave of the 76 CTA pairs to 79 % (``"waves"``), nine to 95 % (the deterministic program).  At
    64 tokens (two K blocks) the partition gives every element two partials and runs; at 192 tokens it would give an
    element more than two (bits that vary run to run), and the row runs the deterministic program."""
    kw = dict(arch="sm_103a", sm_count=153, M=5120, N=256, K=64)
    assert cake_backend._router_rule_spec("nn_t", **kw) == "waves"
    assert det_backend.router_splits("nn_t", 5120, 256, 64, "sm_103a", 153) == 6
    units = cake_backend.router_waves(20, 2, 2, 153)
    assert units is not None
    units = cake_backend.router_units_of(units, 20, 2, 2, 153)
    assert units == 80 and cake_backend.router_at_most_two_partials(20, 2, units)
    assert cake_backend.router_partition("nn_t", **kw) == (units, 0, 0)
    # 192 tokens: a "waves" count, but more than two partials per element
    assert cake_backend.router_waves(20, 2, 6, 153) is not None
    assert cake_backend.router_partition("nn_t", **{**kw, "K": 192}) is None
    monkeypatch.setitem(
        det_backend.ROUTER_SPLIT_RULES, ("sm_103a", 153, "nn_t", 5120), ((1, 192, 9),)
    )
    assert det_backend.router_splits("nn_t", 5120, 256, 64, "sm_103a", 153) == 9
    assert cake_backend.router_partition("nn_t", **kw) is None
    # another SM count keeps the default count (154 SMs: 77 pairs, the last wave 78 % full)
    assert cake_backend.router_partition("nn_t", **{**kw, "sm_count": 154}) is not None


def test_router_units_policy(monkeypatch):
    fwd = dict(M=16231, N=256, K=6144)
    wgrad = dict(M=6144, N=256, K=16231)
    wgrad5120 = dict(M=5120, N=256, K=16231)
    for arch, sms in SM_COUNTS.items():
        # forward: the deterministic plan on every measured row; the input gradient always
        for hidden in (4096, 5120, 6144):
            assert (
                cake_backend.router_units(
                    "kk", arch=arch, sm_count=sms, M=16231, N=256, K=hidden
                )
                is None
            )
        assert cake_backend.router_units("kn", arch=arch, sm_count=sms, **fwd) is None
    # weight gradient at T = 16231 (508 K blocks) and at the token counts of the former head/tail windows: no
    # partition (no table holds a head/tail window); the rows of a "det_s<n>" window run the deterministic program at
    # n splits (test_router_det_split_windows)
    assert cake_backend.ROUTER_RULES[("sm_100a", "nn_t", 6144, 256)] == (
        (1, 256, "simt_n64m64s2"),
        (16129, 16384, "det_s6"),
    )
    assert cake_backend.ROUTER_RULES[("sm_103a", "nn_t", 6144, 256)] == (
        (1, 256, "simt_n64m64s2"),
        (16129, 16384, "det_s6"),
    )
    assert cake_backend.ROUTER_RULES[("sm_100a", "nn_t", 4096, 256)] == (
        (1, 224, "simt_n128m64s2"),
    )
    assert cake_backend.ROUTER_RULES[("sm_103a", "nn_t", 4096, 256)] == (
        (1, 288, "simt_n64m64s2"),
    )
    for arch in ("sm_100a", "sm_103a"):
        for hidden in (4096, 5120, 6144):
            for T in (10240, 16231, 16384, 16385, 16400, 20480, 28672, 49152):
                assert (
                    cake_backend.router_partition(
                        "nn_t", arch=arch, sm_count=152, M=hidden, N=256, K=T
                    )
                    is None
                ), (arch, hidden, T)
    for hidden in (4096, 5120, 6144):
        assert cake_backend.ROUTER_RULES[("sm_107a", "nn_t", hidden, 256)] is None
        for sms in (208, 212, 216):
            part = cake_backend.router_partition(
                "nn_t", arch="sm_107a", sm_count=sms, M=hidden, N=256, K=16231
            )
            if (sms, hidden) == (212, 6144):
                # the K-order window "ksum_S6" (test_router_ksum_window_plans_the_k_order_program): six spans per
                # tile, no head/tail spans
                m_tiles, n_tiles, _ = cake_backend._router_grid(hidden, 256, 16231)
                assert part == (6 * (m_tiles // 2) * n_tiles, 0, 0), part
                continue
            assert part is None, (sms, hidden)
    # the forward of the parts with a table of their own at 16231 tokens: the deterministic plan (the 212-SM forward
    # windows end at 192 tokens)
    for sms in (208, 212):
        for hidden in (4096, 5120, 6144):
            assert (
                cake_backend.router_units(
                    "kk", arch="sm_107a", sm_count=sms, M=16231, N=256, K=hidden
                )
                is None
            )
    # another SM count than the verdicts were measured on, without a table of its own: the deterministic program
    assert cake_backend.ROUTER_SM_COUNT == {
        "sm_100a": 152,
        "sm_103a": 152,
        "sm_107a": 216,
    }
    assert set(cake_backend.ROUTER_RULES_BY_SM) == ROUTER_BY_SM_PARTS
    for sms in (207, 209, 211, 213, 215, 217):
        assert (
            cake_backend.router_units("nn_t", arch="sm_107a", sm_count=sms, **wgrad5120)
            is None
        )
    for arch in ("sm_100a", "sm_103a"):
        for sms in (148, 153):
            assert (
                cake_backend.router_units("nn_t", arch=arch, sm_count=sms, **wgrad5120)
                is None
            )
    # few tokens at hidden 5120 on 153-SM parts (where the SIMT window does not hold): the "waves" window's partition
    # (100 tokens, 4 K blocks: three waves of the 76 CTA pairs, at most one unit per (tile, K block) step -> 160 units
    # over 40 tiles) gives an element four partials, so the row runs the deterministic program (no partition); on
    # 152-SM sm_100a and sm_103a parts the SIMT program (no partition)
    for arch in ("sm_100a", "sm_103a"):
        assert (
            cake_backend._router_rule_spec(
                "nn_t", arch=arch, sm_count=153, M=5120, N=256, K=100
            )
            == "waves"
        )
        assert not cake_backend.router_at_most_two_partials(20, 2, 160)
        assert (
            cake_backend.router_partition(
                "nn_t", arch=arch, sm_count=153, M=5120, N=256, K=100
            )
            is None
        )
        assert (
            cake_backend.router_partition(
                "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=100
            )
            is None
        )
        assert cake_backend.router_simt(
            "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=100
        ) == (64, 64, 2)
    # a row without a verdict (another hidden size or expert count) runs the deterministic program
    assert (
        cake_backend.router_units(
            "nn_t", arch="sm_103a", sm_count=152, M=2048, N=256, K=16231
        )
        is None
    )
    assert (
        cake_backend.router_units(
            "nn_t", arch="sm_103a", sm_count=152, M=5120, N=384, K=16231
        )
        is None
    )
    # explicit splits: S units per tile; the forward takes at most two
    assert (
        cake_backend.router_units("kk", arch="sm_100a", sm_count=148, splits=2, **fwd)
        == 2 * 128
    )
    assert (
        cake_backend.router_units("kk", arch="sm_100a", sm_count=148, splits=1, **fwd)
        == 128
    )
    with pytest.raises(ValueError, match="at most 2"):
        cake_backend.router_units("kk", arch="sm_100a", sm_count=148, splits=3, **fwd)
    with pytest.raises(ValueError, match="splits must be in"):
        cake_backend.router_units(
            "nn_t", arch="sm_100a", sm_count=148, splits=0, **wgrad
        )
    assert (
        cake_backend.router_units(
            "nn_t", arch="sm_100a", sm_count=148, splits=11, **wgrad
        )
        == 11 * 48
    )
    # explicit splits do not depend on the SM count floor
    assert (
        cake_backend.router_units(
            "nn_t", arch="sm_107a", sm_count=212, splits=3, **wgrad
        )
        == 3 * 48
    )
    # verdicts: per (arch, layout, hidden, experts); a forward verdict must keep two partials at most and cannot be
    # a head/tail partition
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_100a", "nn_t", 6144, 256), "pairs"
    )
    assert (
        cake_backend.router_units("nn_t", arch="sm_100a", sm_count=152, **wgrad) == 76
    )
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_100a", "nn_t", 6144, 256), "H0.5x3"
    )
    assert cake_backend.router_partition(
        "nn_t", arch="sm_100a", sm_count=152, **wgrad
    ) == (304, 38, 24 * 508 // 2)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_103a", "kk", 6144, 256), "H0.5x4"
    )
    with pytest.raises(ValueError, match="weight gradient"):
        cake_backend.router_units("kk", arch="sm_103a", sm_count=152, **fwd)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_103a", "kk", 6144, 256), "forward"
    )
    # one unit per CTA pair, at most one per tile (128 tiles here): every tile at most two partials
    assert cake_backend.router_units("kk", arch="sm_103a", sm_count=152, **fwd) == 76
    monkeypatch.setitem(cake_backend.ROUTER_RULES, ("sm_103a", "kk", 6144, 256), "S3")
    with pytest.raises(ValueError, match="more than two partials"):
        cake_backend.router_units("kk", arch="sm_103a", sm_count=152, **fwd)


# the token windows of the router weight gradient (256 experts), (arch, hidden) -> ((first, last, spec), ...); every
# other token count runs the deterministic program.  A SIMT window comes first: where the table's windows do not
# hold, the token count takes the next window that holds it
# fmt: off
ROUTER_WGRAD_WINDOWS = {
    ("sm_100a", 4096): ((1, 224, "simt_n128m64s2"),),
    ("sm_100a", 5120): ((1, 256, "simt_n64m64s2"), (1, 128, "waves")),
    ("sm_100a", 6144): ((1, 256, "simt_n64m64s2"), (16129, 16384, "det_s6")),
    ("sm_103a", 4096): ((1, 288, "simt_n64m64s2"),),
    ("sm_103a", 5120): ((1, 192, "simt_n64m64s2"), (1, 192, "waves")),
    ("sm_103a", 6144): ((1, 256, "simt_n64m64s2"), (16129, 16384, "det_s6")),
}
# the parts with a router table of their own (ROUTER_RULES_BY_SM) and their weight-gradient windows,
# (arch, sm_count, hidden) -> ((first, last, spec), ...)
ROUTER_BY_SM_PARTS = {
    ("sm_100a", 148),
    ("sm_103a", 148),
    ("sm_107a", 208),
    ("sm_107a", 212),
}
# the SIMT windows of 148-SM sm_100a / sm_103a parts (the same on both): the plain 64 x 64 tile in two slices at hidden
# 5120 and 6144, at hidden 4096 the compensated 64 x 64 tile in the Dot2 form at one token and the plain 128 x 64 tile
# from 2 to 224 tokens
_WGRAD_WINDOWS_148 = {
    4096: ((1, 1, "simt_n64m64s2dot2"), (2, 224, "simt_n128m64s2")),
    5120: ((1, 256, "simt_n64m64s2"),),
    6144: ((1, 256, "simt_n64m64s2"),),
}
ROUTER_WGRAD_WINDOWS_BY_SM = {
    **{
        (arch, 148, hidden): windows
        for arch in ("sm_100a", "sm_103a")
        for hidden, windows in _WGRAD_WINDOWS_148.items()
    },
    ("sm_100a", 148, 4096): (*_WGRAD_WINDOWS_148[4096], (16384, 16384, "det_s16")),
    ("sm_100a", 148, 6144): (*_WGRAD_WINDOWS_148[6144], (16129, 16384, "det_s6")),
    ("sm_103a", 148, 6144): (*_WGRAD_WINDOWS_148[6144], (16129, 16384, "det_s6")),
    ("sm_107a", 212, 4096): ((1, 256, "simt_n128m64s2"),),
    ("sm_107a", 212, 5120): ((1, 256, "simt_n128m64s2"), (16172, 16384, "det_s12")),
    ("sm_107a", 212, 6144): ((1, 224, "simt_n128m64s2"), (16172, 16231, "ksum_S6")),
}
# fmt: on
# the input gradient's SIMT windows of 148-SM sm_100a / sm_103a parts (256 experts), (arch, sm_count, hidden) ->
# ((first, last, spec), ...): the 32 x 64 tile in eight slices, on sm_100a at hidden 5120 also the 32 x 64 tile in four
# slices from 64 to 128 tokens; every other part and token count runs the deterministic program
ROUTER_DGRAD_WINDOWS = {
    **{
        (arch, 148, 4096): ((1, 64, "simt_n64m32s8"),)
        for arch in ("sm_100a", "sm_103a")
    },
    ("sm_100a", 148, 5120): ((1, 32, "simt_n64m32s8"), (64, 128, "simt_n64m32s4")),
    ("sm_103a", 148, 5120): ((1, 32, "simt_n64m32s8"),),
    **{
        (arch, 148, 6144): ((1, 32, "simt_n64m32s8"),)
        for arch in ("sm_100a", "sm_103a")
    },
}
# the forward's cluster SIMT windows (256 experts) of 152-SM parts, (arch, hidden) -> ((first, last, spec), ...), and of
# the parts with a router table of their own (212-SM sm_107a), (arch, sm_count, hidden) -> ((first, last, spec), ...);
# every other part, row and token count runs the deterministic program
# fmt: off
_FWD_SM_100A = (
    (1, 8, "simt_c8_kg4_l32_e8t4"), (9, 16, "simt_c8_kg4_l32_e8t16"), (17, 96, "simt_c8_ffma_l16_e16t32"),
    (97, 128, "simt_c8_ffma_l16_e16t64"),
)
_FWD_SM_103A = (
    (1, 16, "simt_c8_kg4_l32_e8t4"), (17, 96, "simt_c8_ffma_l16_e16t32"), (97, 128, "simt_c8_ffma_l16_e16t64"),
    (129, 192, "simt_c8_ffma_l16_e16t32"),
)
ROUTER_FWD_WINDOWS = {
    ("sm_100a", 4096): (*_FWD_SM_100A, (129, 160, "simt_c8_ffma_l16_e16t32")),
    ("sm_100a", 5120): (*_FWD_SM_100A, (129, 192, "simt_c8_ffma_l16_e16t32")),
    ("sm_100a", 6144): (*_FWD_SM_100A, (129, 192, "simt_c8_ffma_l16_e16t32")),
    ("sm_103a", 4096): _FWD_SM_103A,
    ("sm_103a", 5120): _FWD_SM_103A,
    ("sm_103a", 6144): _FWD_SM_103A,
}
ROUTER_FWD_WINDOWS_BY_SM = {
    ("sm_107a", 212, hidden): ((1, 8, "simt_c8_kg4_l32_e8t4"), (9, 192, "simt_c8_ffma_l16_e16t32"))
    for hidden in (4096, 5120, 6144)
}
# fmt: on
# the part each architecture's forward SIMT windows are planned on in the CPU planner tests
ROUTER_FWD_PARTS = {"sm_100a": 152, "sm_103a": 152, "sm_107a": 212}


def _fwd_windows(arch, sm_count, hidden):
    """The forward's SIMT windows of the row ``(arch, "kk", hidden, 256)`` on a part of ``sm_count`` SMs: the part's
    own (:data:`ROUTER_FWD_WINDOWS_BY_SM`) on a part with a router table of its own, :data:`ROUTER_FWD_WINDOWS` on the
    SM count of :data:`ROUTER_RULES` (:data:`ROUTER_SM_COUNT`), none on any other SM count."""
    if (arch, sm_count) in ROUTER_BY_SM_PARTS:
        return ROUTER_FWD_WINDOWS_BY_SM.get((arch, sm_count, hidden), ())
    if sm_count == cake_backend.ROUTER_SM_COUNT[arch]:
        return ROUTER_FWD_WINDOWS.get((arch, hidden), ())
    return ()


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_router_forward_simt_windows(arch):
    """The forward runs its cluster SIMT program only inside its token windows of the table of the part's SM count
    (:data:`ROUTER_RULES` on :data:`ROUTER_SM_COUNT` SMs, the part's own table of :data:`ROUTER_RULES_BY_SM` on 212-SM
    sm_107a parts; not one SM more or fewer, not on the other parts with a table of their own), at the first and last
    token count of each window and not one token past the last; no partition there (:func:`router_partition`); a
    caller's split count, another experts count or hidden size, or ``K`` off a multiple of the cluster's chunk runs no
    SIMT program."""
    measured = cake_backend.ROUTER_SM_COUNT[arch]
    own = {sms for a, sms in ROUTER_BY_SM_PARTS if a == arch}
    for hidden in (4096, 5120, 6144):
        windows = ROUTER_FWD_WINDOWS.get((arch, hidden), ())
        assert cake_backend.ROUTER_RULES[(arch, "kk", hidden, 256)] == (windows or None)
        tokens = {1, 2, 64, 129, 192, 193, 224, 256, 1024, 16231}
        for first, last, _ in windows:
            tokens |= {first, last, last + 1}
        for sms in own:
            by_sm = ROUTER_FWD_WINDOWS_BY_SM.get((arch, sms, hidden), ())
            assert cake_backend.ROUTER_RULES_BY_SM[(arch, sms)][
                (arch, "kk", hidden, 256)
            ] == (by_sm or None)
            for first, last, _ in by_sm:
                tokens |= {first, last, last + 1}
        for sms in sorted(
            {148, 151, 152, 153, 208, 211, 212, 213, 216}
            | {measured - 1, measured, measured + 1}
        ):
            for T in sorted(tokens):
                kw = dict(arch=arch, M=T, N=256, K=hidden, sm_count=sms)
                spec = next(
                    (s for a, b, s in _fwd_windows(arch, sms, hidden) if a <= T <= b),
                    None,
                )
                tile = cake_backend.router_forward_simt(**kw)
                assert tile == cake_backend.router_forward_simt_tile(spec), (
                    hidden,
                    sms,
                    T,
                    tile,
                )
                assert cake_backend.router_partition("kk", **kw) is None, (
                    hidden,
                    sms,
                    T,
                )
                assert cake_backend.router_simt("kk", **kw) is None
                if tile is not None:
                    assert cake_backend.router_forward_simt(splits=1, **kw) is None
                    assert cake_backend.router_forward_simt(**{**kw, "N": 128}) is None
                    assert (
                        cake_backend.router_forward_simt(**{**kw, "K": hidden + 512})
                        is None
                    )


def test_router_forward_simt_specs_and_templates():
    """A forward SIMT spec names the tile ``(kind, lanes, experts, tokens)`` and the program
    ``router_fp32_fwd_simt_c8_<kind>_l<lanes>_e<experts>t<tokens>``; it is no SIMT tile of the gradients and no
    partition spec; every spec of the shipped windows names a program this checkout registers on the window's
    architecture (with the cluster and block of the program's record), and the forward's programs serve no other
    architecture."""
    assert cake_backend.router_forward_simt_tile("simt_c8_kg4_l32_e8t4") == (
        "kg4",
        32,
        8,
        4,
    )
    assert cake_backend.router_forward_simt_tile("simt_c8_ffma_l16_e16t64") == (
        "ffma",
        16,
        16,
        64,
    )
    for spec in (
        "simt_n64m64s2",
        "simt_c8_fma_l16_e16t64",
        "simt_c8_ffma_l16_e16",
        "S2",
        None,
        3,
    ):
        assert cake_backend.router_forward_simt_tile(spec) is None, spec
    assert cake_backend.router_simt_tile("simt_c8_kg4_l32_e8t4") is None
    assert (
        cake_backend.router_forward_simt_template(("ffma", 16, 16, 32))
        == "router_fp32_fwd_simt_c8_ffma_l16_e16t32"
    )
    with pytest.raises(ValueError, match="not a forward SIMT tile"):
        cake_backend.router_forward_simt_template((64, 64, 2))
    with pytest.raises(ValueError, match="unknown router partition spec"):
        cake_backend.router_units_of("simt_c8_kg4_l32_e8t4", 2, 2, 192, 152)
    served = {}
    for (arch, *_), windows in (
        *ROUTER_FWD_WINDOWS.items(),
        *ROUTER_FWD_WINDOWS_BY_SM.items(),
    ):
        for _, _, spec in windows:
            template = cake_backend.router_forward_simt_template(
                cake_backend.router_forward_simt_tile(spec)
            )
            served.setdefault(template, set()).add(arch)
    for template, arches in served.items():
        for arch in cake_jit.ARCHES:
            registered = template in cake_jit.KERNELS.get(arch, {})
            if not cake_jit.KERNELS.get(arch):
                continue  # a checkout without the generated programs
            assert registered == (arch in arches), (template, arch)
            if registered:
                record = cake_jit.MODULES[cake_jit.KERNELS[arch][template]]
                assert record["launch"] == {
                    "block": [256, 1, 1],
                    "cluster": [8, 1, 1],
                }, template
                assert [n for _, n in record["arg_plan"]][:9] == [
                    "X",
                    "W",
                    "out",
                    "T",
                    "N",
                    "K",
                    "ldx",
                    "ldw",
                    "ldo",
                ], template


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_router_forward_simt_plans_its_program(monkeypatch, arch):
    """``plan_atomic_router`` of a forward launch ``X @ W.T`` in a window of the architecture's part
    (:data:`ROUTER_FWD_PARTS`: 152 SMs, 212 SMs on sm_107a): the program of the window's tile over the activations'
    and the router weight's rows in place (``lda`` / ``ldb`` their row strides; strided rows a multiple of 4 elements)
    on ``8 N / experts x ceil(T / tokens)`` CTAs, every element stored once (no part tiles); rows that are not a
    multiple of 4 elements, or a template the checkout does not register, run the deterministic program; the argument
    values of the launch name the program's parameters and keep the call's PDL flag."""
    hidden, N, sms = 5120, 256, ROUTER_FWD_PARTS[arch]
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    X, W, out = torch.zeros(1, hidden), torch.zeros(N, hidden), torch.zeros(1, N)
    plan, reason = cake_backend.plan_atomic_router(
        X, W.t(), out, arch=arch, sm_count=sms
    )
    first_spec = _fwd_windows(arch, sms, hidden)[0][2]
    first_template = cake_backend.router_forward_simt_template(
        cake_backend.router_forward_simt_tile(first_spec)
    )
    assert plan is None and f"{first_template} is not registered" in reason
    for first, last, spec in _fwd_windows(arch, sms, hidden):
        kind, lanes, experts, tokens = cake_backend.router_forward_simt_tile(spec)
        _register(
            monkeypatch,
            arch,
            cake_backend.router_forward_simt_template((kind, lanes, experts, tokens)),
        )
        for T in sorted({first, last}):
            X = torch.zeros(T, hidden + 8)[:, :hidden]
            W = torch.zeros(N, hidden + 4)[:, :hidden]
            out = torch.zeros(T, N)
            plan, reason = cake_backend.plan_atomic_router(
                X, W.t(), out, arch=arch, sm_count=sms
            )
            assert plan is not None, (T, reason)
            assert (
                plan.template
                == f"router_fp32_fwd_simt_c8_{kind}_l{lanes}_e{experts}t{tokens}"
            )
            assert plan.simt == (kind, lanes, experts, tokens) and plan.layout == "kk"
            assert (plan.lda, plan.ldb) == (hidden + 8, hidden + 4)
            assert plan.grid == (8 * N // experts, -(-T // tokens), 1)
            assert not plan.part_tiles and plan.reproducible and plan.units == 0
            values = cake_backend._simt_values(
                plan, A="x", B="w", out="o", ldo=N, enable_pdl=1
            )
            assert {
                k: values[k]
                for k in (
                    "X",
                    "W",
                    "out",
                    "T",
                    "N",
                    "K",
                    "ldx",
                    "ldw",
                    "ldo",
                    "enable_pdl",
                )
            } == dict(
                X="x",
                W="w",
                out="o",
                T=T,
                N=N,
                K=hidden,
                ldx=hidden + 8,
                ldw=hidden + 4,
                ldo=N,
                enable_pdl=1,
            )
            assert (
                cake_backend._simt_values(
                    plan, A="x", B="w", out="o", ldo=N, enable_pdl=0
                )["enable_pdl"]
                == 0
            )
            # rows of the router weight off a multiple of 4 elements: the deterministic program
            W2 = torch.zeros(N, hidden + 2)[:, :hidden]
            plan2, reason2 = cake_backend.plan_atomic_router(
                X, W2.t(), out, arch=arch, sm_count=sms
            )
            assert plan2 is None, (T, reason2)
            # one SM more: the deterministic program
            assert (
                cake_backend.plan_atomic_router(
                    X, W.t(), out, arch=arch, sm_count=sms + 1
                )[0]
                is None
            )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_router_wgrad_token_windows(arch):
    """The weight gradient runs a partition only inside its token windows: at the first and the last token count of a
    window, not one token outside it, and around the few-token "waves" windows; a head/tail or SIMT window only on
    the SM count of its table (:data:`ROUTER_SM_COUNT` for :data:`ROUTER_RULES`, the part's own for
    :data:`ROUTER_RULES_BY_SM`; not one SM more or fewer), a "waves" window on every SM count where
    :func:`router_waves` gives a partition of at most two partials per element (below a SIMT window on the other SM
    counts), a ``"ksum_S<n>"`` window ``n`` units per tile (the K-order program).  A SIMT window runs the SIMT
    program of its tile (:func:`router_simt`), no partition; a ``"det_s<n>"`` window the deterministic program at ``n``
    splits (:func:`router_det_splits`), no partition, on the SM count of its table only.  The deterministic program at
    every other token count and SM count and on every hidden size without a window."""
    measured = cake_backend.ROUTER_SM_COUNT[arch]
    own = {sms for a, sms in ROUTER_BY_SM_PARTS if a == arch}
    assert {sms for a, sms in cake_backend.ROUTER_RULES_BY_SM if a == arch} == own
    sm_counts = {148, 208, 212, 216, measured - 1, measured, measured + 1}
    for sms in own:
        sm_counts |= {sms - 1, sms, sms + 1}
    for hidden in (4096, 5120, 6144):
        main = ROUTER_WGRAD_WINDOWS.get((arch, hidden), ())
        assert cake_backend.ROUTER_RULES[(arch, "nn_t", hidden, 256)] == (main or None)
        tokens = {64, 128, 129, 192, 193, 224, 1024, 8192, 32768, 32769, 65536}
        for first, last, _ in main + sum(
            (ROUTER_WGRAD_WINDOWS_BY_SM.get((arch, sms, hidden), ()) for sms in own),
            (),
        ):
            tokens |= {first - 1, first, last, last + 1}
        m_pairs = hidden // 256
        for sms in sorted(sm_counts):
            if sms in own:
                windows = ROUTER_WGRAD_WINDOWS_BY_SM.get((arch, sms, hidden), ())
                table = cake_backend.ROUTER_RULES_BY_SM[(arch, sms)]
                assert table[(arch, "nn_t", hidden, 256)] == (windows or None)
                assert cake_backend.router_rules(arch, sms) == (table, True)
                holds = True
            else:
                windows, holds = main, sms == measured
                assert cake_backend.router_rules(arch, sms) == (
                    cake_backend.ROUTER_RULES,
                    holds,
                )
            for T in sorted(t for t in tokens if t > 0):
                spec = next(
                    (
                        s
                        for a, b, s in windows
                        if a <= T <= b and (holds or not str(s).startswith("simt_"))
                    ),
                    None,
                )
                k_iters = -(-T // 32)
                got = cake_backend.router_partition(
                    "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=T
                )
                simt = cake_backend.router_simt(
                    "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=T
                )
                if str(spec).startswith("simt_"):
                    assert simt == cake_backend.router_simt_tile(spec), (hidden, T, sms)
                    assert simt is not None
                    want = None
                elif spec == "waves":
                    units = cake_backend.router_waves(m_pairs, 2, k_iters, sms)
                    if units is not None:
                        units = cake_backend.router_units_of(
                            units, m_pairs, 2, k_iters, sms
                        )
                        if not cake_backend.router_at_most_two_partials(
                            m_pairs, 2, units
                        ):
                            units = None  # bits that vary: the deterministic program
                    want = None if units is None else (units, 0, 0)
                elif (
                    spec is None
                    or not holds
                    or cake_backend.router_det_spec_splits(spec) is not None
                ):
                    want = None
                else:
                    ksum = cake_backend.router_ksum_spec_splits(spec)
                    want = cake_backend.router_partition_of(
                        spec if ksum is None else f"S{ksum}", m_pairs, 2, k_iters, sms
                    )
                assert got == want, (hidden, T, sms)
                det_n = cake_backend.router_det_splits(
                    "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=T
                )
                want_n = cake_backend.router_det_spec_splits(spec) if holds else None
                assert det_n == want_n, (hidden, T, sms)
                if not str(spec).startswith("simt_"):
                    assert simt is None, (hidden, T, sms)
    if arch == "sm_100a":
        # hidden 4096 at 10240 tokens (320 K blocks): the deterministic plan (no head/tail window); the smallest
        # measured "waves" count on 152 SMs runs the SIMT program
        assert (
            cake_backend.router_partition(
                "nn_t", arch=arch, sm_count=152, M=4096, N=256, K=10240
            )
            is None
        )
        assert (
            cake_backend.router_partition(
                "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=64
            )
            is None
        )
        assert cake_backend.router_simt(
            "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=64
        ) == (64, 64, 2)
    if arch == "sm_103a":
        # hidden 5120: the SIMT program up to 192 tokens on 152 SMs, the "waves" window under it on 153 SMs, where
        # the SIMT window does not hold (64 tokens: 80 units, two partials per element; 192 tokens: 228 units, three
        # waves of the 76 CTA pairs and more than two partials per element: the deterministic program)
        for K in (64, 192):
            assert (
                cake_backend.router_partition(
                    "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=K
                )
                is None
            )
            assert cake_backend.router_simt(
                "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=K
            ) == (64, 64, 2)
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=153, M=5120, N=256, K=64
        ) == (80, 0, 0)
        assert (
            cake_backend.router_partition(
                "nn_t", arch=arch, sm_count=153, M=5120, N=256, K=192
            )
            is None
        )


def _plan_router_row(layout, arch, sm_count, hidden, T, experts=256):
    """``plan_atomic_router`` of the weight gradient (``nn_t``: ``G.t() @ X``) or the input gradient (``kn``: ``G @
    W``) of a router of ``hidden`` columns and ``experts`` experts at ``T`` tokens on a part of ``sm_count`` SMs."""
    G = torch.zeros(T, experts)
    if layout == "nn_t":
        args = (G.t(), torch.zeros(T, hidden), torch.zeros(experts, hidden))
    else:
        args = (G, torch.zeros(experts, hidden), torch.zeros(T, hidden))
    return cake_backend.plan_atomic_router(*args, arch=arch, sm_count=sm_count)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_router_148_sm_simt_windows_plan_their_registered_programs(arch):
    """On a 148-SM part each SIMT window of its table (:data:`ROUTER_WGRAD_WINDOWS_BY_SM`, :data:`ROUTER_DGRAD_WINDOWS`)
    plans, at its first, middle and last token count, the program of its tile (plain, or compensated with its form),
    which this checkout registers for ``arch``.  The token counts between two windows of a row and one token past a
    row's last window take no SIMT program (the deterministic program; the ``"det_s<n>"`` windows of the weight
    gradient: ``test_router_det_split_windows``)."""
    windows_seen = planned = 0
    for hidden in (4096, 5120, 6144):
        simt_wgrad = tuple(
            w
            for w in ROUTER_WGRAD_WINDOWS_BY_SM[(arch, 148, hidden)]
            if cake_backend.router_det_spec_splits(w[2]) is None
        )
        for layout, windows in (
            ("nn_t", simt_wgrad),
            ("kn", ROUTER_DGRAD_WINDOWS[(arch, 148, hidden)]),
        ):
            for first, last, spec in windows:
                windows_seen += 1
                tile = cake_backend.router_simt_tile(spec)
                assert tile is not None, spec
                template = cake_backend.router_simt_template(tile, layout)
                assert template in cake_jit.KERNELS[arch], (spec, template)
                for T in sorted({first, (first + last) // 2, last}):
                    plan, reason = _plan_router_row(layout, arch, 148, hidden, T)
                    assert plan is not None, (layout, hidden, T, reason)
                    assert plan.simt == tile, (layout, hidden, T, plan.simt)
                    assert plan.template == template, (layout, hidden, T)
                    planned += 1
            for (_, last, _), (first, _, _) in zip(windows, windows[1:], strict=False):
                for T in range(last + 1, first):
                    plan, _ = _plan_router_row(layout, arch, 148, hidden, T)
                    assert plan is None, (layout, hidden, T)
            past = windows[-1][1] + 1
            plan, _ = _plan_router_row(layout, arch, 148, hidden, past)
            assert plan is None, (layout, hidden, past)
    # the weight gradient's two windows at hidden 4096 (the first a single token) and one at 5120 and 6144; the input
    # gradient's one window per hidden size, two at hidden 5120 on sm_100a
    assert (windows_seen, planned) == ((8, 22) if arch == "sm_100a" else (7, 19))


def test_router_ksum_window_plans_the_k_order_program(monkeypatch):
    """A ``"ksum_S<n>"`` window of the weight gradient (212-SM sm_107a, hidden 6144 at 16172 to 16231 tokens) plans
    the K-order program of the part (:data:`ROUTER_KSUM_TEMPLATES_BY_SM`): ``n`` units per tile, every element stored
    once (no part tiles, no fill of ``out``), a reproducible plan, the slots and tickets of
    :func:`router_ksum_workspace`.  One token outside the window plans no K-order program; without its template
    registered the row runs the deterministic program."""
    arch, sms, hidden = "sm_107a", 212, 6144
    template = cake_backend.ROUTER_KSUM_TEMPLATES_BY_SM[(arch, sms)]
    assert template == "router_fp32_gemm_ts_r6p6_atomic_nn_t_ksum_ht_c2_nreg"
    assert cake_backend.router_ksum_template(arch, sms) == template
    for other in (208, 211, 213, 216):
        assert cake_backend.router_ksum_template(arch, other) is None
    assert cake_backend.router_ksum_spec_splits("ksum_S6") == 6
    for spec in ("S6", "det_s6", "waves", "ksum_S", "H0.5x4", 6, None):
        assert cake_backend.router_ksum_spec_splits(spec) is None, spec
    assert template in cake_jit.KERNELS[arch]
    for T in (16171, 16172, 16200, 16231, 16232):
        plan, reason = _plan_router_row("nn_t", arch, sms, hidden, T)
        if not 16172 <= T <= 16231:
            assert plan is None or not plan.ksum, T
            continue
        assert plan is not None, (T, reason)
        tiles = plan.m_pairs * plan.n_tiles
        assert (plan.template, plan.ksum, plan.units) == (template, True, 6 * tiles)
        assert (plan.m_pairs, plan.n_tiles) == (24, 2)
        assert plan.reproducible and not plan.part_tiles and plan.head_spans == 0
        slots, words = plan.ksum_workspace
        assert (slots, words) == cake_backend.router_ksum_workspace(24, 2, 6 * tiles)
        assert (slots, words) == ((6 * 24 + 23) * 2 * 256 * 128, 24 * 2 * 512)
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=T
        ) == (6 * tiles, 0, 0)
    # without the program registered: the deterministic program
    table = {t: m for t, m in cake_jit.KERNELS[arch].items() if t != template}
    monkeypatch.setitem(cake_jit.KERNELS, arch, table)
    plan, reason = _plan_router_row("nn_t", arch, sms, hidden, 16172)
    assert plan is None and "K-order" in reason


def test_k2_ksum_binding_carves_its_workspace(monkeypatch):
    """A cached K-order launch binds the arena's slots and tickets (``k2_ksum_slots`` / ``k2_ksum_tickets``, the sizes
    of :attr:`CakeDenseGemmNondetRouterPlan.ksum_workspace`) into the program's ``ws`` / ``counters`` arguments: with a
    call's views in place its kernel argument list is the list the launch orders per call from ``_k2_values``; no fill
    of ``out``."""
    monkeypatch.setattr(cake_backend.tvm_ffi, "from_dlpack", lambda t, **_: t)
    monkeypatch.setattr(cake_backend, "_CU_DRIVER", False)
    arch, sms, T, H = "sm_107a", 212, 16172, 6144
    X, G, out = torch.empty(T, H), torch.empty(T, 256), torch.empty(256, H)
    A, B = G.t(), X
    plan, reason = cake_backend.plan_atomic_router(A, B, out, arch=arch, sm_count=sms)
    assert plan is not None and plan.ksum, reason
    module_name = cake_jit.select_module(arch, plan.template)
    arg_plan = cake_jit.arg_plan_of(cake_jit.MODULES[module_name])
    names = [n for _, n in arg_plan]
    assert "ws" in names and "counters" in names and "D" not in names, names
    a_view, split_src = cake_backend._k2_views(A, B, plan)
    split_module = det_backend._fused_split_for(arch, split_src)
    split_arg_plan = (
        None
        if split_module is None
        else det_backend.arg_plan_of(det_backend.MODULES[split_module])
    )
    bound = cake_backend._k2_binding(
        plan, out, split_src, module_name, arg_plan, split_module, split_arg_plan, 0
    )
    assert bound.ksum_workspace == plan.ksum_workspace and bound.fill is None
    slots, words = plan.ksum_workspace
    n = split_src.numel()
    arena = cake_backend._AtomicArena(torch.device("cpu"), (0, 0))
    arena.require("k2_stack", 3 * n)
    if split_module is None:
        arena.require("k2_split_scratch", n)
    arena.require("k2_ksum_slots", slots)
    arena.require("k2_ksum_tickets", words)
    bound.carve(arena, split_module is None)
    assert bound.version == arena.version
    parts = arena.buffers["k2_stack"][: 3 * n].view(3, *split_src.shape)
    out32 = out if bound.out_contiguous else det_backend.flat_alias(out)
    kernel_args, _ = bound.arguments((a_view, out32, out, split_src))
    values = cake_backend._k2_values(
        plan,
        A=a_view,
        parts=parts,
        out=out,
        ws=arena.buffers["k2_ksum_slots"][:slots],
        counters=arena.buffers["k2_ksum_tickets"][:words],
    )
    _same_arguments(
        kernel_args,
        cake_backend._order_atomic_arguments(module_name, arg_plan, values, plan.grid),
    )


def test_router_per_sm_count_tables_decide_only_their_own_sm_count(monkeypatch):
    """A part whose SM count has a table in :data:`ROUTER_RULES_BY_SM` takes that table only (every row of
    :data:`ROUTER_RULES` of its architecture has an entry there, and rows measured on that SM count only may too; no
    ``"waves"`` window, closed head/tail, SIMT, ``"det_s<n>"`` or ``"ksum_S<n>"`` windows of the weight gradient, closed
    SIMT windows of the input gradient and closed cluster SIMT windows of the forward); one SM more or fewer runs the deterministic
    program, and the measured SM count of :data:`ROUTER_RULES` decides as before."""
    for (arch, sms), table in cake_backend.ROUTER_RULES_BY_SM.items():
        assert sms != cake_backend.ROUTER_SM_COUNT[arch]
        assert set(table) >= {k for k in cake_backend.ROUTER_RULES if k[0] == arch}
        for (key_arch, layout, _hidden, _experts), rule in table.items():
            assert key_arch == arch
            if rule is None:
                continue
            assert layout in ("kk", "nn_t", "kn") and isinstance(rule, tuple)
            for first, last, spec in rule:
                assert 1 <= first <= last <= 32768
                if layout == "kk":
                    assert cake_backend.router_forward_simt_tile(spec) is not None
                    continue
                if layout == "kn":
                    assert cake_backend.router_simt_tile(spec) is not None
                    continue
                assert (
                    cake_backend.router_head_tail(spec) is not None
                    or cake_backend.router_simt_tile(spec) is not None
                    or cake_backend.router_det_spec_splits(spec) is not None
                    or cake_backend.router_ksum_spec_splits(spec) is not None
                )
    wgrad = dict(M=6144, N=256, K=16231)
    key = ("sm_107a", "nn_t", 6144, 256)
    tables = dict(cake_backend.ROUTER_RULES_BY_SM)
    tables[("sm_107a", 210)] = {key: ((16172, 16384, "H0.5x3"),)}
    monkeypatch.setattr(cake_backend, "ROUTER_RULES_BY_SM", tables)
    # 105 CTA pairs: 104 head units (a multiple of the two column tiles), 3 * 105 tail units rounded down to 314
    assert cake_backend.router_partition(
        "nn_t", arch="sm_107a", sm_count=210, **wgrad
    ) == (104 + 314, 52, 24 * 508 // 2)
    for sms in (209, 211, 216):
        assert (
            cake_backend.router_partition("nn_t", arch="sm_107a", sm_count=sms, **wgrad)
            is None
        )
    # a row without an entry in the part's table runs the deterministic program, whatever ROUTER_RULES says
    monkeypatch.setitem(cake_backend.ROUTER_RULES, key, "H0.5x4")
    tables[("sm_107a", 210)] = {}
    assert (
        cake_backend.router_partition("nn_t", arch="sm_107a", sm_count=210, **wgrad)
        is None
    )
    assert cake_backend.router_partition(
        "nn_t", arch="sm_107a", sm_count=216, **wgrad
    ) == (540, 54, 24 * 508 // 2)


def test_router_simt_specs_name_their_tiles_and_templates():
    """A SIMT window spec ``"simt_n<tile_n>m<tile_m>s<slices>"`` names its tile and the template of its program on
    a weight-gradient row (``router_fp32_gemm_simt_n<tile_n>m<tile_m>s<slices>``) and on an input-gradient row
    (``router_fp32_dgrad_simt_m<tile_m>n<tile_n>s<slices>``); a compensated one ``"simt_n<tile_n>m<tile_m>s<slices>
    <form>"`` (``g<group>`` or ``dot2``) names the tile with its form and the compensated program
    (``router_fp32_gemm_simt_kahan_...g<group>``, without the suffix for ``g1``;
    ``router_fp32_gemm_simt_kahan_dot2_...``; ``router_fp32_dgrad_simt_kahan_...g<group>``); every other spec is none,
    the Dot2 form is the weight gradient's only, and the forward has no SIMT program."""
    assert cake_backend.router_simt_tile("simt_n64m64s4") == (64, 64, 4)
    assert cake_backend.router_simt_tile("simt_n128m64s2") == (128, 64, 2)
    assert cake_backend.router_simt_tile("simt_n64m64s2g4") == (64, 64, 2, "g4")
    assert cake_backend.router_simt_tile("simt_n128m64s2g4") == (128, 64, 2, "g4")
    assert cake_backend.router_simt_tile("simt_n64m64s2g1") == (64, 64, 2, "g1")
    assert cake_backend.router_simt_tile("simt_n64m64s2dot2") == (64, 64, 2, "dot2")
    assert cake_backend.router_simt_tile("simt_n64m32s4g4") == (64, 32, 4, "g4")
    for spec in (
        None,
        3,
        "waves",
        "H0.5x3",
        "S2",
        "simt",
        "simt_n64m64",
        "xsimt_n64m64s4",
        "simt_n64m64s2g",
        "simt_n64m64s2g4x",
        "simt_n64m64g4",
        "simt_n64m64s2dot",
        "simt_n64m64s2dot3",
        "simt_n64m64s2g4dot2",
    ):
        assert cake_backend.router_simt_tile(spec) is None, spec
    # the templates of the compensated forms
    for spec, layout, template in (
        ("simt_n64m64s2dot2", "nn_t", "router_fp32_gemm_simt_kahan_dot2_n64m64s2"),
        ("simt_n64m64s2g1", "nn_t", "router_fp32_gemm_simt_kahan_n64m64s2"),
        ("simt_n64m64s2g4", "nn_t", "router_fp32_gemm_simt_kahan_n64m64s2g4"),
        ("simt_n128m64s2g4", "nn_t", "router_fp32_gemm_simt_kahan_n128m64s2g4"),
        ("simt_n64m32s4g4", "kn", "router_fp32_dgrad_simt_kahan_m32n64s4g4"),
    ):
        tile = cake_backend.router_simt_tile(spec)
        assert cake_backend.router_simt_template(tile, layout) == template, spec
        prefix = cake_jit.TEMPLATE_PREFIXES[3 if layout == "nn_t" else 4]
        assert template.startswith(prefix), spec
    with pytest.raises(ValueError, match="not a SIMT tile"):
        cake_backend.router_simt_template((64, 64, 2, "g4", 1))
    for form in (4, "g0", "g", "dot", "x4"):
        with pytest.raises(ValueError, match="not a SIMT tile"):
            cake_backend.router_simt_template((64, 64, 2, form))
    with pytest.raises(ValueError, match="no SIMT program"):
        cake_backend.router_simt_template((64, 64, 2, "g4"), "kk")
    with pytest.raises(ValueError, match="the Dot2 form is the weight gradient's"):
        cake_backend.router_simt_template((64, 32, 4, "dot2"), "kn")
    assert (
        cake_backend.router_simt_template((128, 64, 2))
        == "router_fp32_gemm_simt_n128m64s2"
    )
    assert cake_backend.router_simt_template((64, 64, 4)).startswith(
        cake_jit.TEMPLATE_PREFIXES
    )
    assert (
        cake_backend.router_simt_template((64, 32, 4), "kn")
        == "router_fp32_dgrad_simt_m32n64s4"
    )
    assert cake_backend.router_simt_template(
        cake_backend.router_simt_tile("simt_n64m64s4"), "kn"
    ) == ("router_fp32_dgrad_simt_m64n64s4")
    assert cake_backend.router_simt_template((64, 32, 8), "kn").startswith(
        cake_jit.TEMPLATE_PREFIXES[4]
    )
    with pytest.raises(ValueError, match="no SIMT program"):
        cake_backend.router_simt_template((64, 64, 4), "kk")


def test_router_simt_windows_decide_the_weight_gradient(monkeypatch):
    """A SIMT window of the weight gradient: :func:`router_simt` gives its tile at every token count of the window,
    on the SM count of the table only (:data:`ROUTER_SM_COUNT`, or a part's own table), when the tile divides the
    hidden size and the experts; :func:`router_partition` gives no partition there (the launch is no stream-K launch).
    A caller's split count, the forward and every token count outside the window take the stream-K planner."""
    arch, hidden, N = "sm_103a", 5120, 256
    key = (arch, "nn_t", hidden, N)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES,
        key,
        (
            (1, 128, "simt_n64m64s2"),
            (129, 256, "simt_n64m64s4"),
            (16172, 32768, "H0.5x3"),
        ),
    )
    kw = dict(arch=arch, M=hidden, N=N)
    for T, want in (
        (1, (64, 64, 2)),
        (64, (64, 64, 2)),
        (128, (64, 64, 2)),
        (129, (64, 64, 4)),
        (256, (64, 64, 4)),
    ):
        assert cake_backend.router_simt("nn_t", sm_count=152, K=T, **kw) == want, T
        assert cake_backend.router_partition("nn_t", sm_count=152, K=T, **kw) is None, T
        # another SM count without a table of its own (148-SM sm_103a parts have one): no SIMT window (the
        # deterministic program)
        for sms in (151, 153, 160):
            assert cake_backend.router_simt("nn_t", sm_count=sms, K=T, **kw) is None
            assert (
                cake_backend.router_partition("nn_t", sm_count=sms, K=T, **kw) is None
            )
        # a caller's split count runs the stream-K program
        assert (
            cake_backend.router_simt("nn_t", sm_count=152, K=T, splits=1, **kw) is None
        )
        assert cake_backend.router_partition(
            "nn_t", sm_count=152, K=T, splits=1, **kw
        ) == (
            hidden // 256 * 2,
            0,
            0,
        )
    for T in (257, 1024, 16171):
        assert cake_backend.router_simt("nn_t", sm_count=152, K=T, **kw) is None, T
        assert cake_backend.router_partition("nn_t", sm_count=152, K=T, **kw) is None, T
    assert cake_backend.router_simt("nn_t", sm_count=152, K=16172, **kw) is None
    assert (
        cake_backend.router_partition("nn_t", sm_count=152, K=16172, **kw) is not None
    )
    # the forward never takes a SIMT window
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, (arch, "kk", hidden, N), "simt_n64m64s4"
    )
    assert (
        cake_backend.router_simt("kk", arch=arch, sm_count=152, M=64, N=N, K=hidden)
        is None
    )
    # a tile that does not divide the hidden size: no SIMT program, and no partition either
    hidden_odd = dict(arch=arch, M=5184, N=N, K=64)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, (arch, "nn_t", 5184, N), "simt_n64m128s2"
    )
    assert cake_backend.router_simt("nn_t", sm_count=152, **hidden_odd) is None
    assert cake_backend.router_partition("nn_t", sm_count=152, **hidden_odd) is None
    # a SIMT window before a "waves" window: on the table's SM count the SIMT program; on another SM count the
    # "waves" partition wherever router_waves gives one of at most two partials per element (as without the SIMT
    # window)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, key, ((1, 256, "simt_n64m64s2"), (1, 128, "waves"))
    )
    m_pairs, waves_seen = hidden // 256, 0
    for sms in (150, 152, 153, 160):
        for T in (64, 128, 129, 256):
            simt = cake_backend.router_simt("nn_t", sm_count=sms, K=T, **kw)
            part = cake_backend.router_partition("nn_t", sm_count=sms, K=T, **kw)
            if sms == 152:
                assert simt == (64, 64, 2) and part is None, (sms, T)
                continue
            assert simt is None, (sms, T)
            k_iters = -(-T // 32)
            units = None
            if T <= 128:
                units = cake_backend.router_waves(m_pairs, 2, k_iters, sms)
            if units is not None:
                units = cake_backend.router_units_of(units, m_pairs, 2, k_iters, sms)
                if cake_backend.router_at_most_two_partials(m_pairs, 2, units):
                    waves_seen += 1
                else:
                    units = None  # bits that vary: the deterministic program
            assert part == (None if units is None else (units, 0, 0)), (sms, T)
    assert waves_seen
    # a part with a table of its own takes its own windows only
    tables = dict(cake_backend.ROUTER_RULES_BY_SM)
    tables[("sm_107a", 210)] = {
        ("sm_107a", "nn_t", 6144, N): ((1, 256, "simt_n128m64s2"),)
    }
    monkeypatch.setattr(cake_backend, "ROUTER_RULES_BY_SM", tables)
    wgrad = dict(arch="sm_107a", M=6144, N=N, K=200)
    assert cake_backend.router_simt("nn_t", sm_count=210, **wgrad) == (128, 64, 2)
    for sms in (209, 211, 216):
        assert cake_backend.router_simt("nn_t", sm_count=sms, **wgrad) is None


def test_k2_planner_plans_the_simt_router(monkeypatch):
    """In a SIMT window the weight gradient's plan runs the SIMT program of the window's tile: the swapped launch
    (``M`` hidden, ``N`` experts, ``K`` tokens) over one CTA per tile of the transposed ``out``, the operands' row
    strides bound, no stack, no part tiles, every element stored once.  Views the program cannot read through
    whole rows of a multiple of 4 elements, and a template this checkout does not register, run the deterministic
    program."""
    arch, hidden, N, T = "sm_100a", 4096, 256, 231
    kw = dict(arch=arch, sm_count=152)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES,
        (arch, "nn_t", hidden, N),
        ((1, 256, "simt_n128m64s2"),),
    )
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    X, G, out = torch.zeros(T, hidden), torch.zeros(T, N), torch.zeros(N, hidden)
    plan, reason = cake_backend.plan_atomic_router(G.t(), X, out, **kw)
    assert (
        plan is None and "router_fp32_gemm_simt_n128m64s2 is not registered" in reason
    )
    _register(monkeypatch, arch, "router_fp32_gemm_simt_n128m64s2")
    plan, reason = cake_backend.plan_atomic_router(G.t(), X, out, **kw)
    assert plan is not None, reason
    assert (
        plan.template,
        plan.simt,
        plan.layout,
        plan.swapped,
        plan.a_mn,
        plan.b_mn,
    ) == (
        "router_fp32_gemm_simt_n128m64s2",
        (128, 64, 2),
        "nn_t",
        True,
        True,
        True,
    )
    assert (plan.M, plan.N, plan.K, plan.lda, plan.ldb) == (hidden, N, T, hidden, N)
    # 64 x 2 CTAs, ceil(231 / 16) K blocks of 16 rows
    assert (plan.m_tiles, plan.n_tiles, plan.k_iters, plan.units) == (64, 2, 15, 0)
    assert plan.grid == (64, 2, 1)
    assert not plan.part_tiles and plan.reproducible
    assert plan.head_spans == 0 and plan.head_steps == 0
    # the values the program takes: the operands' and out's row strides, the launch's dimensions, and no
    # programmatic dependent launch whatever the call's PDL flag
    values = cake_backend._simt_values(
        plan, A="x", B="g", out="o", ldo=hidden + 4, enable_pdl=1
    )
    assert values == dict(
        A="x",
        B="g",
        out="o",
        M=hidden,
        N=N,
        K=T,
        lda=hidden,
        ldb=N,
        ldo=hidden + 4,
        enable_pdl=0,
    )
    # padded rows (multiples of 4 elements) are read in place
    Xp, Gp = torch.zeros(T, hidden + 8)[:, :hidden], torch.zeros(T, N + 4)[:, :N]
    plan, _ = cake_backend.plan_atomic_router(Gp.t(), Xp, out, **kw)
    assert (plan.template, plan.lda, plan.ldb) == (
        "router_fp32_gemm_simt_n128m64s2",
        hidden + 8,
        N + 4,
    )
    # output-gradient rows that are not a multiple of 4 elements: the deterministic program
    G2 = torch.zeros(T, N + 2)[:, :N]
    plan, reason = cake_backend.plan_atomic_router(G2.t(), X, out, **kw)
    assert plan is None and "output gradient's rows" in reason
    # stride-0 operands (the views of a broadcast): the deterministic program
    G0, X0 = torch.zeros(1, N).expand(T, N), torch.zeros(1, hidden).expand(T, hidden)
    plan, reason = cake_backend.plan_atomic_router(G0.t(), X0, out, **kw)
    assert plan is None and "SIMT" in reason
    # a caller's split count runs the stream-K program at the window's token counts
    _register(monkeypatch, arch, "router_fp32_gemm_atomic_nn_t")
    plan, _ = cake_backend.plan_atomic_router(G.t(), X, out, splits=1, **kw)
    assert plan.template == "router_fp32_gemm_atomic_nn_t" and plan.simt is None
    # past the window: the deterministic program (no verdict)
    X3, G3 = torch.zeros(257, hidden), torch.zeros(257, N)
    assert cake_backend.plan_atomic_router(G3.t(), X3, out, **kw)[0] is None


#: The input gradient's SIMT windows as shipped, ``(arch, SM count, hidden) -> ((first, last, tile), ...)`` (256
#: experts): 152-SM sm_100a and sm_103a parts (:data:`ROUTER_RULES`), 212-SM sm_107a parts and 148-SM sm_100a and
#: sm_103a parts (:data:`ROUTER_RULES_BY_SM`, :data:`ROUTER_DGRAD_WINDOWS`); every other part and row runs the
#: deterministic program.
DGRAD_SIMT_WINDOWS = {
    **{(arch, 148, 4096): ((1, 64, (64, 32, 8)),) for arch in ("sm_100a", "sm_103a")},
    ("sm_100a", 148, 5120): ((1, 32, (64, 32, 8)), (64, 128, (64, 32, 4))),
    ("sm_103a", 148, 5120): ((1, 32, (64, 32, 8)),),
    **{(arch, 148, 6144): ((1, 32, (64, 32, 8)),) for arch in ("sm_100a", "sm_103a")},
    ("sm_100a", 152, 4096): ((1, 128, (64, 32, 4)),),
    ("sm_100a", 152, 5120): ((1, 128, (64, 32, 4)),),
    ("sm_100a", 152, 6144): ((1, 64, (64, 32, 4)),),
    ("sm_103a", 152, 4096): ((1, 128, (64, 32, 4)),),
    ("sm_103a", 152, 5120): ((1, 128, (64, 32, 4)),),
    ("sm_103a", 152, 6144): ((1, 64, (64, 32, 4)),),
    ("sm_107a", 212, 4096): ((1, 64, (64, 32, 8)), (65, 128, (64, 64, 4))),
    ("sm_107a", 212, 5120): ((1, 64, (64, 32, 8)), (65, 128, (64, 64, 4))),
    ("sm_107a", 212, 6144): ((1, 64, (64, 32, 8)),),
}


def test_router_simt_windows_of_the_input_gradient(monkeypatch):
    """The input gradient (``kn``: ``M`` tokens, ``N`` hidden, ``K`` experts) takes the SIMT program of its window at
    every token count of the window, on the SM count of its table only; one token past the window, a caller's split
    count, another SM count or architecture, another expert count and a hidden size the tile does not divide run the
    deterministic program, and ``router_partition`` gives none (the input gradient has no stream-K program).  The
    shipped windows include those of 148-SM sm_100a / sm_103a parts, where the token counts between two windows of a
    row run the deterministic program too."""
    for key, rows in ROUTER_DGRAD_WINDOWS.items():
        assert DGRAD_SIMT_WINDOWS[key] == tuple(
            (lo, hi, cake_backend.router_simt_tile(spec)) for lo, hi, spec in rows
        ), key
    seen = 0
    tokens = (1, 17, 32, 63, 64, 65, 96, 127, 128, 129, 192, 256, 1000, 16231)
    for arch in cake_jit.ARCHES:
        for sms in ROUTER_SM_FACTS:
            for hidden in (4096, 5120, 6144):
                windows = DGRAD_SIMT_WINDOWS.get((arch, sms, hidden), ())
                for T in tokens:
                    want = next((t for lo, hi, t in windows if lo <= T <= hi), None)
                    kw = dict(arch=arch, sm_count=sms, M=T, N=hidden, K=256)
                    assert cake_backend.router_simt("kn", **kw) == want, (
                        arch,
                        sms,
                        hidden,
                        T,
                    )
                    assert cake_backend.router_simt("kn", splits=1, **kw) is None
                    assert cake_backend.router_partition("kn", **kw) is None
                    seen += want is not None
                    other = dict(kw, K=128)
                    assert cake_backend.router_simt("kn", **other) is None
    # nine token counts from 1 to 128 at hidden 4096 and 5120, five from 1 to 64 at hidden 6144, on each 152-SM
    # sm_100a / sm_103a part and the 212-SM sm_107a part; on each 148-SM sm_100a / sm_103a part five from 1 to 64 at
    # hidden 4096 and three from 1 to 32 at hidden 5120 and 6144, and on sm_100a at hidden 5120 five more from 64 to 128
    assert seen == 3 * (9 + 9 + 5) + 2 * (5 + 3 + 3) + 5
    # a hidden size the tile does not divide: the deterministic program
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES,
        ("sm_100a", "kn", 5160, 256),
        ((1, 128, "simt_n64m32s4"),),
    )
    kw = dict(arch="sm_100a", sm_count=152, M=64, K=256)
    assert cake_backend.router_simt("kn", N=5160, **kw) is None
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES,
        ("sm_100a", "kn", 5184, 256),
        ((1, 128, "simt_n64m32s4"),),
    )
    assert cake_backend.router_simt("kn", N=5184, **kw) == (64, 32, 4)


def test_k2_planner_plans_the_simt_input_gradient(monkeypatch):
    """In a SIMT window the input gradient's plan (``kn``: ``A`` the output gradient ``G[T, E]``, ``B`` the router
    weight ``W[E, H]``) runs the SIMT program of the window's tile over ``G`` and ``W`` as they are: one CTA per
    ``tile_m x tile_n`` tile of ``out``, ``n_tiles x m_tiles`` CTAs with the last row tile partial, the operands' row
    strides bound, no stack, no part tiles, every element stored once.  Rows of ``G`` or ``W`` the program cannot read
    whole (``E`` or a row stride not a multiple of 4 elements), a template this checkout does not register, a
    caller's split count and a token count past the window run the deterministic program."""
    arch, hidden, E = "sm_100a", 5120, 256
    kw = dict(arch=arch, sm_count=152)
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    W = torch.zeros(E, hidden)
    G, out = torch.zeros(17, E), torch.zeros(17, hidden)
    plan, reason = cake_backend.plan_atomic_router(G, W, out, **kw)
    assert (
        plan is None and "router_fp32_dgrad_simt_m32n64s4 is not registered" in reason
    )
    _register(monkeypatch, arch, "router_fp32_dgrad_simt_m32n64s4")
    for T in (1, 17, 32, 33, 100, 128):
        G, out = torch.zeros(T, E), torch.zeros(T, hidden)
        plan, reason = cake_backend.plan_atomic_router(G, W, out, **kw)
        assert plan is not None, (T, reason)
        assert (
            plan.template,
            plan.simt,
            plan.layout,
            plan.swapped,
            plan.a_mn,
            plan.b_mn,
        ) == ("router_fp32_dgrad_simt_m32n64s4", (64, 32, 4), "kn", False, False, True)
        assert (plan.M, plan.N, plan.K, plan.lda, plan.ldb) == (T, hidden, E, E, hidden)
        m_tiles = -(-T // 32)
        assert (plan.m_tiles, plan.n_tiles, plan.k_iters, plan.units) == (
            m_tiles,
            hidden // 64,
            E // 16,
            0,
        )
        assert plan.grid == (hidden // 64, m_tiles, 1)
        assert not plan.part_tiles and plan.reproducible
        assert plan.head_spans == 0 and plan.head_steps == 0
    # the values the program takes: the operands' and out's row strides, the launch's dimensions, and no
    # programmatic dependent launch whatever the call's PDL flag
    values = cake_backend._simt_values(
        plan, A="g", B="w", out="o", ldo=hidden + 4, enable_pdl=1
    )
    assert values == dict(
        A="g",
        B="w",
        out="o",
        M=128,
        N=hidden,
        K=E,
        lda=E,
        ldb=hidden,
        ldo=hidden + 4,
        enable_pdl=0,
    )
    G, out = torch.zeros(64, E), torch.zeros(64, hidden)
    # padded rows (multiples of 4 elements) are read in place
    Gp, Wp = torch.zeros(64, E + 4)[:, :E], torch.zeros(E, hidden + 8)[:, :hidden]
    outp = torch.zeros(64, hidden + 4)[:, :hidden]
    plan, reason = cake_backend.plan_atomic_router(Gp, Wp, outp, **kw)
    assert plan is not None, reason
    assert (plan.template, plan.lda, plan.ldb) == (
        "router_fp32_dgrad_simt_m32n64s4",
        E + 4,
        hidden + 8,
    )
    # router-weight rows that are not a multiple of 4 elements: the deterministic program
    W2 = torch.zeros(E, hidden + 2)[:, :hidden]
    plan, reason = cake_backend.plan_atomic_router(G, W2, out, **kw)
    assert plan is None and "router weight's rows" in reason
    # an expert count not a multiple of 4 (a row with a window forced for it): the deterministic program
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES,
        (arch, "kn", hidden, 254),
        ((1, 128, "simt_n64m32s4"),),
    )
    G3, W3 = torch.zeros(64, E)[:, :254], torch.zeros(254, hidden)
    plan, reason = cake_backend.plan_atomic_router(G3, W3, out, **kw)
    assert plan is None and "K a multiple of 4" in reason
    # stride-0 rows of the output gradient (the view of a broadcast): the deterministic program
    G0 = torch.zeros(1, E).expand(64, E)
    plan, reason = cake_backend.plan_atomic_router(G0, W, out, **kw)
    assert plan is None and "SIMT" in reason
    # a caller's split count, and a token count past the window: the deterministic program
    plan, reason = cake_backend.plan_atomic_router(G, W, out, splits=1, **kw)
    assert plan is None and "kn class runs the deterministic program" in reason
    G4, out4 = torch.zeros(129, E), torch.zeros(129, hidden)
    plan, reason = cake_backend.plan_atomic_router(G4, W, out4, **kw)
    assert plan is None and "kn class runs the deterministic program" in reason
    # another part (a 148-SM sm_103a part has no input-gradient window)
    plan, reason = cake_backend.plan_atomic_router(
        G, W, out, arch="sm_103a", sm_count=148
    )
    assert plan is None and "kn class runs the deterministic program" in reason


def test_router_no_head_tail_windows():
    """No router table routes the weight gradient to the head/tail program (its windows varied from run to run and
    missed the accuracy check): no spec of :data:`ROUTER_RULES` or :data:`ROUTER_RULES_BY_SM` starts with "H", and at
    the token counts of the former windows (sm_103a hidden 4096 20480 to 49152 and hidden 6144 16385 to 20480 tokens,
    hidden 5120 10240 to 32768 tokens) no 148- to 153-SM sm_100a / sm_103a part and no sm_107a part partitions the
    weight gradient, but for the K-order window of 212-SM parts (hidden 6144, 16172 to 16231 tokens: ``"ksum_S6"``,
    spans of equal length, no head/tail spans)."""
    tables = [cake_backend.ROUTER_RULES, *cake_backend.ROUTER_RULES_BY_SM.values()]
    assert not [
        spec
        for table in tables
        for rule in table.values()
        if isinstance(rule, tuple)
        for *_, spec in rule
        if str(spec).startswith("H")
    ]

    def part(arch, sms, hidden, T):
        return cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=T
        )

    cases = (
        (4096, 20480),
        (4096, 49152),
        (5120, 10240),
        (5120, 16172),
        (5120, 16384),
        (5120, 28672),
        (5120, 32768),
        (6144, 16385),
        (6144, 20480),
    )
    for arch in ("sm_100a", "sm_103a"):
        for sms in (148, 151, 152, 153):
            for hidden, T in cases:
                assert part(arch, sms, hidden, T) is None, (arch, sms, hidden, T)
    for sms in (208, 212, 216):
        for hidden in (4096, 5120, 6144):
            for T in (16172, 16231, 16384):
                got = part("sm_107a", sms, hidden, T)
                if (sms, hidden) == (212, 6144) and T <= 16231:
                    assert got is not None and got[1:] == (0, 0), (sms, hidden, T, got)
                    continue
                assert got is None, (sms, hidden, T)


def test_router_wgrad_at_the_deterministic_split_rule_rows():
    """On 148- and 152-SM sm_100a / sm_103a parts the weight gradient (256 experts) at the token counts of the
    deterministic package's ``ROUTER_SPLIT_RULES`` rows (16129 to 16384 tokens: nine splits at hidden 5120, three at
    hidden 6144) runs the deterministic program and no partition: at hidden 5120 at the deterministic package's nine
    splits (no window), at hidden 6144 at six splits (its "det_s6" window: three splits missed the accuracy check)."""
    seen = set()
    for arch in ("sm_100a", "sm_103a"):
        for sms in (148, 152):
            for hidden in (5120, 6144):
                rule = det_backend.ROUTER_SPLIT_RULES.get((arch, sms, "nn_t", hidden))
                assert rule, (arch, sms, hidden)
                for lo, hi, splits in rule:
                    for T in range(lo, hi + 1):
                        case = (arch, sms, hidden, T)
                        det_n = det_backend.router_splits(
                            "nn_t", hidden, 256, T, arch, sms
                        )
                        assert det_n == splits, case
                        seen.add((arch, sms, hidden, splits))
                        kw = dict(arch=arch, sm_count=sms, M=hidden, N=256, K=T)
                        assert cake_backend.router_partition("nn_t", **kw) is None, case
                        got = cake_backend.router_det_splits("nn_t", **kw)
                        assert got == (6 if hidden == 6144 else None), case
    assert seen == {
        (arch, sms, hidden, 9 if hidden == 5120 else 3)
        for arch in ("sm_100a", "sm_103a")
        for sms in (148, 152)
        for hidden in (5120, 6144)
    }


def test_k2_planner_plans_the_stream_k_router(monkeypatch):
    arch, T, K, N = "sm_103a", 4099, 6144, 256
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    fwd = (X, W.t(), torch.zeros(T, N))
    wgrad = (G.t(), X, torch.zeros(N, K))
    kw = dict(arch=arch, sm_count=152)
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    plan, reason = cake_backend.plan_atomic_router(*fwd, **kw)
    assert plan is None and "not registered" in reason
    _register(
        monkeypatch, arch, "router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"
    )
    # the forward: no verdict ships (the deterministic plan); an explicit split count runs this package's program
    plan, reason = cake_backend.plan_atomic_router(*fwd, **kw)
    assert plan is None and "planner" in reason
    plan, _ = cake_backend.plan_atomic_router(*fwd, splits=1, **kw)
    assert (plan.layout, plan.template, plan.swapped) == (
        "kk",
        "router_fp32_gemm_atomic_kk",
        False,
    )
    assert (plan.m_tiles, plan.n_tiles, plan.k_iters, plan.tiles) == (34, 2, 192, 34)
    # one unit per tile: stored whole
    assert plan.units == 34 and plan.grid == (68, 1, 1)
    assert not plan.part_tiles and plan.reproducible
    plan, _ = cake_backend.plan_atomic_router(*fwd, splits=2, **kw)
    assert plan.units == 68 and plan.part_tiles and plan.reproducible
    # the weight gradient of hidden 6144: the deterministic plan
    assert cake_backend.plan_atomic_router(*wgrad, **kw)[0] is None
    # a "waves" verdict at hidden 5120 (40 tiles, 129 K blocks: 3 waves of the 76 pairs, more than two partials per
    # element): the deterministic program
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_103a", "nn_t", 5120, 256), "waves"
    )
    X5 = torch.zeros(T, 5120)
    plan, reason = cake_backend.plan_atomic_router(
        G.t(), X5, torch.zeros(N, 5120), **kw
    )
    assert plan is None and "planner" in reason
    # an "S3" verdict (three units per tile, set here; no shipped table holds one): three partials per element
    monkeypatch.setitem(cake_backend.ROUTER_RULES, ("sm_103a", "nn_t", 5120, 256), "S3")
    plan, _ = cake_backend.plan_atomic_router(G.t(), X5, torch.zeros(N, 5120), **kw)
    assert (plan.layout, plan.template, plan.swapped) == (
        "nn_t",
        "router_fp32_gemm_atomic_nn_t",
        True,
    )
    assert (plan.M, plan.N, plan.K, plan.k_iters) == (5120, N, T, 129)
    assert plan.units == 3 * 40 and plan.part_tiles and not plan.reproducible
    dgrad = (G, W, torch.zeros(T, K))
    assert cake_backend.plan_atomic_router(*dgrad, **kw)[0] is None
    assert plan.head_spans == 0 and plan.head_steps == 0


def test_k2_planner_plans_the_head_tail_router(monkeypatch):
    """A head/tail verdict (``"H0.5x4"`` on sm_107a, set here for hidden 4096, 5120 and 6144; sm_100a / sm_103a at
    hidden 5120) runs the head/tail program of the architecture: ``router_fp32_gemm_atomic_nn_t_ht_c3_nreg`` (on
    sm_103a ``router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg``) with the partition's ``head_spans`` /
    ``head_steps``, which the launch passes to the program; without that template registered the row runs the
    deterministic program.  The shipped tables hold no head/tail window (those windows varied from run to run and
    missed the accuracy check); the former windows are set here (the programs stay registered)."""
    arch, T, N = "sm_107a", 16231, 256
    kw = dict(arch=arch, sm_count=216)
    tables = {
        part: dict(rows) for part, rows in cake_backend.ROUTER_RULES_BY_SM.items()
    }
    for sms, hidden in ((212, 5120), (212, 6144), (208, 5120)):
        tables[(arch, sms)][(arch, "nn_t", hidden, 256)] = ((16172, 16384, "H0.5x4"),)
    monkeypatch.setattr(cake_backend, "ROUTER_RULES_BY_SM", tables)
    for gb in ("sm_100a", "sm_103a"):
        monkeypatch.setitem(
            cake_backend.ROUTER_RULES,
            (gb, "nn_t", 5120, 256),
            ((16172, 16384, "H0.5x3"),),
        )
    for hidden in (4096, 5120, 6144):
        monkeypatch.setitem(
            cake_backend.ROUTER_RULES, (arch, "nn_t", hidden, 256), "H0.5x4"
        )
    G = torch.zeros(T, N)
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    _register(
        monkeypatch, arch, "router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"
    )
    X5 = torch.zeros(T, 5120)
    plan, reason = cake_backend.plan_atomic_router(
        G.t(), X5, torch.zeros(N, 5120), **kw
    )
    assert plan is None and reason == (
        "router_fp32_gemm_atomic_nn_t_ht_c3_nreg is not registered for sm_107a"
    )
    _register(monkeypatch, arch, "router_fp32_gemm_atomic_nn_t_ht_c3_nreg")
    for hidden, m_pairs in ((4096, 16), (5120, 20), (6144, 24)):
        X = torch.zeros(T, hidden)
        out = torch.zeros(N, hidden)
        plan, _ = cake_backend.plan_atomic_router(G.t(), X, out, **kw)
        assert (plan.layout, plan.template, plan.swapped) == (
            "nn_t",
            "router_fp32_gemm_atomic_nn_t_ht_c3_nreg",
            True,
        )
        assert (plan.M, plan.N, plan.K, plan.m_tiles, plan.n_tiles, plan.k_iters) == (
            hidden,
            N,
            T,
            2 * m_pairs,
            2,
            508,
        )
        assert (plan.units, plan.head_spans, plan.head_steps) == (
            540,
            54,
            m_pairs * 254,
        )
        assert plan.grid == (1080, 1, 1) and plan.part_tiles and not plan.reproducible
        values = cake_backend._k2_values(plan, A="a", parts="b", out=out)
        assert (values["units"], values["head_spans"], values["head_steps"]) == (
            540,
            54,
            m_pairs * 254,
        )
    # an explicit split count runs the stream-K program with spans alike
    plan, _ = cake_backend.plan_atomic_router(
        G.t(), X5, torch.zeros(N, 5120), splits=3, **kw
    )
    assert plan.template == "router_fp32_gemm_atomic_nn_t" and plan.head_spans == 0
    # fewer SMs without a table of their own: the deterministic program
    plan, reason = cake_backend.plan_atomic_router(
        G.t(), X5, torch.zeros(N, 5120), arch=arch, sm_count=214
    )
    assert plan is None and "planner" in reason
    # 212 and 208 SMs: the tables of their own (ROUTER_RULES_BY_SM), hidden 5120 "H0.5x4" on 106 / 104 CTA pairs and,
    # on 212 SMs, hidden 6144 "H0.5x4" too; 212 SMs run the head/tail program with the A parts in tensor memory
    # (ROUTER_HEAD_TAIL_TEMPLATES_BY_SM), and the deterministic program while this checkout does not register it
    ts = "router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg"
    plan, reason = cake_backend.plan_atomic_router(
        G.t(), X5, torch.zeros(N, 5120), arch=arch, sm_count=212
    )
    assert plan is None and reason == f"{ts} is not registered for sm_107a"
    _register(monkeypatch, arch, ts)
    for sms, pairs, template in (
        (212, 106, ts),
        (208, 104, "router_fp32_gemm_atomic_nn_t_ht_c3_nreg"),
    ):
        for hidden, m_pairs in ((5120, 20), (6144, 24)):
            plan, reason = cake_backend.plan_atomic_router(
                G.t(),
                torch.zeros(T, hidden),
                torch.zeros(N, hidden),
                arch=arch,
                sm_count=sms,
            )
            if (sms, hidden) == (208, 6144):
                assert plan is None and "planner" in reason
                continue
            assert plan.template == template, (sms, hidden)
            assert (plan.units, plan.head_spans, plan.head_steps) == (
                5 * pairs,
                pairs // 2,
                m_pairs * 254,
            )
            assert plan.grid == (10 * pairs, 1, 1)
        X4 = torch.zeros(T, 4096)
        plan, reason = cake_backend.plan_atomic_router(
            G.t(), X4, torch.zeros(N, 4096), arch=arch, sm_count=sms
        )
        assert plan is None and "planner" in reason
    # 216 SMs keep the three-chunk-buffer head/tail program
    plan, _ = cake_backend.plan_atomic_router(G.t(), X5, torch.zeros(N, 5120), **kw)
    assert plan.template == "router_fp32_gemm_atomic_nn_t_ht_c3_nreg"
    # sm_100a / sm_103a, hidden 5120: three tail units per pair (76 head + 228 tail units on 152 SMs); sm_103a runs
    # the head/tail program with the A parts in tensor memory, and the deterministic program while this checkout does
    # not register it
    templates = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES
    assert templates == {"sm_103a": ts}
    by_sm = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES_BY_SM
    assert by_sm == {("sm_107a", 212): ts}
    for gb, template in (
        ("sm_100a", "router_fp32_gemm_atomic_nn_t_ht_c3_nreg"),
        ("sm_103a", ts),
    ):
        assert cake_backend.router_head_tail_template(gb) == template
        monkeypatch.setitem(cake_jit.KERNELS, gb, {})
        _register(
            monkeypatch,
            gb,
            "router_fp32_gemm_atomic_kk",
            "router_fp32_gemm_atomic_nn_t",
            "router_fp32_gemm_atomic_nn_t_ht_c3_nreg",
        )
        if gb == "sm_103a":
            plan, reason = cake_backend.plan_atomic_router(
                G.t(), X5, torch.zeros(N, 5120), arch=gb, sm_count=152
            )
            assert plan is None and reason == f"{ts} is not registered for sm_103a"
            _register(monkeypatch, gb, ts)
        plan, _ = cake_backend.plan_atomic_router(
            G.t(), X5, torch.zeros(N, 5120), arch=gb, sm_count=152
        )
        assert (plan.template, plan.units, plan.head_spans, plan.head_steps) == (
            template,
            304,
            38,
            20 * 254,
        )
        assert plan.grid == (608, 1, 1) and plan.part_tiles


#: The SM counts of the parts the router tables were measured on (GB200 / GB300: 148 and 152; R200: 208, 212, 216).
ROUTER_SM_FACTS = (148, 152, 208, 212, 216)


def test_k2_head_tail_program_of_each_architecture_and_sm_count():
    """With this checkout's registry, on every architecture and the SM counts of the measured parts, the head/tail
    program of the weight gradient is the one with the A parts in tensor memory on sm_103a and on 212-SM sm_107a parts
    (``ROUTER_HEAD_TAIL_TEMPLATES_BY_SM``) and ``router_fp32_gemm_atomic_nn_t_ht_c3_nreg`` elsewhere; the program is
    registered for sm_103a and sm_107a.  The shipped router tables hold no head/tail window (those windows varied from
    run to run and missed the accuracy check), so no weight-gradient plan (256 experts, hidden 4096 / 5120 / 6144, token
    counts at and around the former windows) partitions the launch."""
    ts = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES["sm_103a"]
    c3 = "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX
    assert [a for a, table in cake_jit.KERNELS.items() if ts in table] == [
        "sm_103a",
        "sm_107a",
    ]
    assert cake_jit.MODULES[cake_jit.select_module("sm_107a", ts)]["arches"] == [
        "sm_103a",
        "sm_107a",
    ]
    for arch in cake_jit.ARCHES:
        for sms in (None, *ROUTER_SM_FACTS):
            want = ts if arch == "sm_103a" or (arch, sms) == ("sm_107a", 212) else c3
            assert cake_backend.router_head_tail_template(arch, sms) == want, (
                arch,
                sms,
            )
    tokens = (8192, 10240, 12288, 12289, 16171, 16172, 16231, 16232, 16384, 16385)
    tokens += (20479, 20480, 24576, 24577, 28672, 28673, 32768, 32769, 49152, 49153)
    N = 256
    for arch in cake_jit.ARCHES:
        for sms in ROUTER_SM_FACTS:
            for hidden in (4096, 5120, 6144):
                out = torch.empty(N, hidden)
                for T in tokens:
                    # stride-0 views of the gradient [T, N] and the activations [T, hidden]: the layout classes of
                    # G.t() and X without their memory
                    G = torch.empty(1, N).expand(T, N)
                    X = torch.empty(1, hidden).expand(T, hidden)
                    plan, _ = cake_backend.plan_atomic_router(
                        G.t(), X, out, arch=arch, sm_count=sms
                    )
                    assert plan is None or not plan.head_spans, (arch, sms, hidden, T)


def test_k2_values_feed_every_registered_router_program():
    """The keyword arguments of a K2 launch name every argument of every registered stream-K router program's
    argument plan (the head/tail program's ``head_spans`` / ``head_steps`` included); those of a SIMT launch every
    argument of every registered SIMT program's plan (the weight gradient's and the input gradient's), the operands'
    row strides and the launch flag before the grid, 0 whatever the call's PDL flag (a gradient's SIMT program never
    launches as a programmatic dependent launch); those of a forward SIMT launch every argument of every registered
    forward program's plan, with the call's PDL flag."""
    out = torch.zeros(256, 5120)
    plan = cake_backend.CakeDenseGemmNondetRouterPlan(
        M=5120, N=256, K=16231, a_mn=True, b_mn=True, swapped=True, layout="nn_t", m_tiles=40,
        n_tiles=2, k_iters=508, units=540, template="router_fp32_gemm_atomic_nn_t_ht_c3_nreg",
        head_spans=54, head_steps=5080,
    )  # fmt: skip
    values = cake_backend._k2_values(plan, A="a", parts="b", out=out)
    simt_prefix, dgrad_prefix = cake_jit.TEMPLATE_PREFIXES[3:5]
    simt_plan = dataclasses.replace(
        plan,
        template="router_fp32_gemm_simt_n64m64s2",
        m_tiles=80,
        n_tiles=4,
        k_iters=1015,
        units=0,
        head_spans=0,
        head_steps=0,
        simt=(64, 64, 2),
        lda=5120,
        ldb=256,
    )
    simt_values = cake_backend._simt_values(
        simt_plan, A="x", B="g", out="o", ldo=5120, enable_pdl=1
    )
    assert simt_values == cake_backend._simt_values(
        simt_plan, A="x", B="g", out="o", ldo=5120, enable_pdl=0
    )
    for name, record in cake_jit.MODULES.items():
        if not record["template"].startswith(simt_prefix):
            continue
        plan_args = cake_jit.arg_plan_of(record)
        arguments = cake_backend._order_atomic_arguments(
            name, plan_args, simt_values, simt_plan.grid
        )
        assert [a for _, a in plan_args] == [
            "A", "B", "out", "M", "N", "K", "lda", "ldb", "ldo", "enable_pdl",
            "grid_x", "grid_y", "grid_z",
        ]  # fmt: skip
        assert arguments == [
            "x",
            "g",
            "o",
            5120,
            256,
            16231,
            5120,
            256,
            5120,
            0,
            80,
            4,
            1,
        ]
    # the input gradient's: out[M, N] = A[M, K] @ B[K, N] over n_tiles x m_tiles CTAs, the last row tile partial
    dgrad_plan = dataclasses.replace(
        simt_plan,
        template="router_fp32_dgrad_simt_m32n64s4",
        M=231,
        N=5120,
        K=256,
        a_mn=False,
        b_mn=True,
        swapped=False,
        layout="kn",
        m_tiles=8,
        n_tiles=80,
        k_iters=16,
        simt=(64, 32, 4),
        lda=260,
        ldb=5124,
    )
    assert dgrad_plan.grid == (80, 8, 1)
    dgrad_values = cake_backend._simt_values(
        dgrad_plan, A="g", B="w", out="o", ldo=5128, enable_pdl=1
    )
    dgrad_names = [
        n for n, r in cake_jit.MODULES.items() if r["template"].startswith(dgrad_prefix)
    ]
    assert dgrad_names
    for name in dgrad_names:
        plan_args = cake_jit.arg_plan_of(cake_jit.MODULES[name])
        arguments = cake_backend._order_atomic_arguments(
            name, plan_args, dgrad_values, dgrad_plan.grid
        )
        assert [a for _, a in plan_args] == [
            "A", "B", "out", "M", "N", "K", "lda", "ldb", "ldo", "enable_pdl",
            "grid_x", "grid_y", "grid_z",
        ]  # fmt: skip
        assert arguments == [
            "g",
            "w",
            "o",
            231,
            5120,
            256,
            260,
            5124,
            5128,
            0,
            80,
            8,
            1,
        ]
    # the forward's: out[T, N] = X[T, K] @ W[N, K].T over 8 N / experts x ceil(T / tokens) CTAs in clusters of 8,
    # with the call's PDL flag
    fwd_prefix = cake_jit.TEMPLATE_PREFIXES[6]
    fwd_plan = dataclasses.replace(
        simt_plan,
        template="router_fp32_fwd_simt_c8_ffma_l16_e16t32",
        M=100,
        N=256,
        K=5120,
        a_mn=False,
        b_mn=False,
        swapped=False,
        layout="kk",
        m_tiles=4,
        n_tiles=16,
        k_iters=8,
        simt=("ffma", 16, 16, 32),
        lda=5124,
        ldb=5128,
    )
    assert fwd_plan.grid == (128, 4, 1)
    fwd_names = [
        n for n, r in cake_jit.MODULES.items() if r["template"].startswith(fwd_prefix)
    ]
    assert fwd_names
    for pdl in (0, 1):
        fwd_values = cake_backend._simt_values(
            fwd_plan, A="x", B="w", out="o", ldo=260, enable_pdl=pdl
        )
        for name in fwd_names:
            plan_args = cake_jit.arg_plan_of(cake_jit.MODULES[name])
            arguments = cake_backend._order_atomic_arguments(
                name, plan_args, fwd_values, fwd_plan.grid
            )
            assert [a for _, a in plan_args] == [
                "X", "W", "out", "T", "N", "K", "ldx", "ldw", "ldo", "enable_pdl",
                "grid_x", "grid_y", "grid_z",
            ]  # fmt: skip
            assert arguments == [
                "x", "w", "o", 100, 256, 5120, 5124, 5128, 260, pdl, 128, 4, 1,
            ]  # fmt: skip
    # the stream-K router programs (the partial sum program has its own plan: test_partials_sum_serves_...)
    names = [
        n
        for n, r in cake_jit.MODULES.items()
        if r["template"].startswith("router_")
        and not r["template"].startswith((simt_prefix, dgrad_prefix, fwd_prefix))
        and r["template"] != cake_backend.PARTIALS_SUM_TEMPLATE
    ]
    ksum_templates = set(cake_backend.ROUTER_KSUM_TEMPLATES_BY_SM.values())
    assert ksum_templates
    # the K-order program takes its workspace slots and tickets instead of D (the binding of a K-order plan passes
    # them: test_k2_ksum_binding_carves_its_workspace)
    ksum_values = cake_backend._k2_values(
        dataclasses.replace(plan, ksum=True, head_spans=0, head_steps=0),
        A="a",
        parts="b",
        out=out,
        ws="ws",
        counters="tickets",
    )
    for name in names:
        plan_args = cake_jit.arg_plan_of(cake_jit.MODULES[name])
        ksum = cake_jit.MODULES[name]["template"] in ksum_templates
        arguments = cake_backend._order_atomic_arguments(
            name, plan_args, ksum_values if ksum else values, plan.grid
        )
        assert len(arguments) == len(plan_args), name
        names_of = [a for _, a in plan_args]
        takes_ws = {"ws", "counters"} <= set(names_of) and "D" not in names_of
        assert takes_ws == ksum, name
        if cake_jit.MODULES[name]["template"].endswith(
            cake_backend.ROUTER_HEAD_TAIL_SUFFIX
        ):
            assert [a for _, a in plan_args][-6:] == [
                "head_spans",
                "head_steps",
                "enable_pdl",
                "grid_x",
                "grid_y",
                "grid_z",
            ]
        if (
            cake_jit.MODULES[name]["template"]
            in cake_backend.ROUTER_HEAD_TAIL_TEMPLATES.values()
        ):
            # the head/tail program with the A parts in tensor memory also takes the launch flag (0 here)
            assert [a for _, a in plan_args][-6:] == [
                "head_spans",
                "head_steps",
                "enable_pdl",
                "grid_x",
                "grid_y",
                "grid_z",
            ]
            assert arguments[-4] == 0


def test_k2_out_fill_is_one_driver_memset_with_a_zero_fallback(monkeypatch):
    """The fill of a K2 ``out`` with part tiles: one ``cuMemsetD32Async`` over contiguous rows, one
    ``cuMemsetD2D32Async`` (pitch in bytes) over padded rows, on the stream it is given; ``out.zero_()`` when torch's
    current device is another device, when the driver refuses the call, or without ``cuda.bindings``."""
    calls = []

    class Driver:
        class CUresult:
            CUDA_SUCCESS = 0

        status = 0

        def cuMemsetD32Async(self, ptr, value, count, stream):
            calls.append(("1d", ptr, value, count, stream))
            return (self.status,)

        def cuMemsetD2D32Async(self, ptr, pitch, value, width, height, stream):
            calls.append(("2d", ptr, pitch, value, width, height, stream))
            return (self.status,)

    driver = Driver()
    monkeypatch.setattr(cake_backend, "_CU_DRIVER", driver)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    dense = torch.full((6, 8), 7.0)
    padded = torch.full((6, 12), 7.0)[:, :8]
    cake_backend._out_fill(dense, 1)(1234)
    cake_backend._out_fill(padded, 1)(99)
    assert calls == [
        ("1d", dense.data_ptr(), 0, 48, 1234),
        ("2d", padded.data_ptr(), 48, 0, 8, 6, 99),
    ]
    assert bool((dense == 7).all()) and bool((padded == 7).all())
    calls.clear()
    cake_backend._out_fill(dense, 0)(1234)  # another current device
    assert calls == [] and bool((dense == 0).all())
    driver.status = 1  # the driver refuses the memset
    cake_backend._out_fill(padded, 1)(99)
    assert len(calls) == 1 and bool((padded == 0).all())
    base = torch.full((6, 12), 7.0)
    monkeypatch.setattr(cake_backend, "_CU_DRIVER", False)  # no cuda.bindings
    cake_backend._out_fill(base[:, :8], 1)(99)
    assert len(calls) == 1 and bool((base[:, :8] == 0).all())
    assert bool((base[:, 8:] == 7).all())


def _geometry(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.stride(), t.storage_offset(), t.data_ptr(), t.dtype)


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("a_mn", [False, True])
@pytest.mark.parametrize("b_mn", [False, True])
def test_k2_views_are_the_views_of_the_swapped_operands(swapped, a_mn, b_mn):
    """``_k2_views`` takes the streamed operand's span and the split's source in at most two view operations of the
    caller's tensors: the same sizes, strides, storage offset and data pointer as the views of the swapped operands
    ``A_eff = B.t()``, ``B_eff = A.t()`` (so the launch binds the same arguments)."""
    plan = type("Plan", (), dict(swapped=swapped, a_mn=a_mn, b_mn=b_mn))()
    for A, B in (
        (torch.empty(64, 96), torch.empty(96, 32)),
        (torch.empty(96, 64).t(), torch.empty(32, 96).t()),
        (torch.empty(64, 100)[:, 4:], torch.empty(40, 104).t()[8:]),
    ):
        A_eff, B_eff = (B.t(), A.t()) if swapped else (A, B)
        want_src = B_eff if b_mn else B_eff.t()
        want_a = A_eff.t().unsqueeze(0) if a_mn else A_eff.unsqueeze(0)
        a_view, split_src = cake_backend._k2_views(A, B, plan)
        assert _geometry(a_view) == _geometry(want_a)
        assert _geometry(split_src) == _geometry(want_src)


# The router entry points' own tensors as launch views: (A, B) of the K2 GEMM and the views the entry point passes
# fmt: off
ENTRY_POINT_VIEWS = {
    "fwd": lambda X, W, G: ((X, W.t()), (X.unsqueeze(0), W, cake_backend._FORWARD_VIEWS)),
    "dgrad": lambda X, W, G: ((G, W), (G.unsqueeze(0), W, cake_backend._DGRAD_VIEWS)),
    "wgrad": lambda X, W, G: ((G.t(), X), (X.unsqueeze(0), G, cake_backend._WGRAD_VIEWS)),
}
# fmt: on


@pytest.mark.parametrize("op", sorted(ENTRY_POINT_VIEWS))
def test_entry_point_views_are_the_launch_views(monkeypatch, op):
    """An entry point's own tensors are the launch views of the deterministic package's plan (its fail-closed views)
    and of this package's plan whenever the plan has the entry point's flags (the forward's and the weight
    gradient's stream-K plan, the input gradient's SIMT plan in its window), so the launch binds the same
    arguments; contiguous operands always have them, other layouts fall back to ``_k2_views``."""
    arch, sm_count, T, H = "sm_103a", 152, 64, 5120
    _register(
        monkeypatch, arch, "router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"
    )
    _register(monkeypatch, "sm_100a", "router_fp32_dgrad_simt_m32n64s4")
    # the input gradient's SIMT window: a 152-SM sm_100a part; the others' stream-K program with two splits
    atomic_kw = {
        "dgrad": dict(arch="sm_100a", sm_count=152),
        "fwd": dict(arch=arch, sm_count=sm_count, splits=2),
        "wgrad": dict(arch=arch, sm_count=sm_count, splits=2),
    }[op]
    # fmt: off
    layouts = (
        (torch.empty(T, H), torch.empty(256, H), torch.empty(T, 256), True),
        (torch.empty(T, H + 64)[:, :H], torch.empty(256, H), torch.empty(T, 272)[:, :256], True),
        (torch.empty(H, T).t(), torch.empty(H, 256).t(), torch.empty(256, T).t(), False),
    )
    # fmt: on
    used = 0
    for X, W, G, canonical in layouts:
        (A, B), views = ENTRY_POINT_VIEWS[op](X, W, G)
        out = torch.empty(A.shape[0], B.shape[1])
        plans = []
        try:
            plan, A_eff, B_eff = det_backend.cake_dense_gemm_det_train_router_plan(
                A, B, out
            )
        except (ValueError, NotImplementedError):
            assert not canonical
        else:
            want = det_backend._k2_checked_views(A, B, A_eff, B_eff, plan)
            plans.append((plan, want))
        plan, _ = cake_backend.plan_atomic_router(A, B, out, **atomic_kw)
        if plan is not None:
            assert (plan.simt is not None) == (op == "dgrad"), op
            plans.append((plan, cake_backend._k2_views(A, B, plan)))
        assert plans or not canonical
        for plan, (want_a, want_src) in plans:
            flags = (plan.swapped, plan.a_mn, plan.b_mn)
            assert flags == views[2] or not canonical, (op, flags)
            used += flags == views[2]
            a_view, split_src = cake_backend._launch_views(A, B, plan, views)
            assert _geometry(a_view) == _geometry(want_a), (op, flags)
            assert _geometry(split_src) == _geometry(want_src), (op, flags)
    assert used


class _FakeModule:
    """A loaded module stand-in: every entry is one recorder (CPU binding tests: nothing is loaded)."""

    def __init__(self):
        self.entry = Recorder()

    def __getattr__(self, name):
        return self.entry


@pytest.mark.parametrize(
    "case",
    [
        "wgrad",
        "fwd",
        "dgrad",
        "dgrad_padded_out",
        "wgrad_padded_out",
        "torch_chain",
    ],
)
@pytest.mark.parametrize("pdl", [0, 1])
def test_splitk_binding_is_the_deterministic_launch(monkeypatch, case, pdl):
    """The binding of the deterministic program on this package's convenience path (``_plan_splitk_k2``) orders the
    deterministic package's arguments once: with a call's views in place its kernel and fused-split argument lists are
    the lists the deterministic package's convenience path resolves (``_k2_values`` / ``_split_values`` in its
    argument plans) -- the stack, the partials ``[splits, rows, cols]`` of several splits or ``out`` itself (its flat
    alias when padded) for one split -- carved from this package's arena, and carved again after the arena grows; the
    PDL flag is bound into both lists.  Several splits bind this package's partial sum: the partials, the call's
    ``out`` (its flat alias when padded), the split count, the rows, columns and row stride of ``out``, the PDL flag and
    one thread per four columns of a row."""
    monkeypatch.setattr(cake_backend.tvm_ffi, "from_dlpack", lambda t, **_: t)
    arch, T, H = "sm_103a", 64, 6144
    monkeypatch.setattr(det_backend, "require_arch", lambda device: arch)
    monkeypatch.setattr(
        det_backend,
        "load_cake_dense_projection_det_gemm_module",
        lambda name, arch: _FakeModule(),
    )
    monkeypatch.setattr(
        cake_backend,
        "load_cake_dense_projection_gemm_module",
        lambda name, arch: _FakeModule(),
    )
    monkeypatch.setattr(cake_backend, "_ATOMIC_ARENAS", {})
    X, W = torch.empty(T, H), torch.empty(256, H)
    G = torch.empty(T, 272)[:, :256] if case == "torch_chain" else torch.empty(T, 256)
    op = case.split("_")[0] if case != "torch_chain" else "wgrad"
    (A, B), views = ENTRY_POINT_VIEWS[op](X, W, G)
    rows, cols = A.shape[0], B.shape[1]
    out = (
        torch.empty(rows, cols + 64)[:, :cols]
        if case.endswith("padded_out")
        else torch.empty(rows, cols)
    )
    splits = 1 if op == "dgrad" else None
    entry = cake_backend._plan_splitk_k2(A, B, out, splits, 0, 0, pdl)
    plan, bound, arena = entry.plan, entry.k2, entry.arena
    assert bound is not None and arena is not None and entry.entry is not None
    assert (
        plan
        == det_backend.cake_dense_gemm_det_train_router_plan(A, B, out, splits=splits)[
            0
        ]
    )
    assert plan.splits == {"fwd": 4, "dgrad": 1, "wgrad": 6}[op]
    a_view, split_src = cake_backend._launch_views(A, B, plan, views)
    split_module = det_backend._fused_split_for(arch, split_src)
    assert (split_module is None) == (case == "torch_chain")
    assert (entry.split_entry is None) == (split_module is None)
    n = split_src.numel()
    for grow in (1, 4):
        if grow > 1:
            arena.require("k2_stack", 3 * n * grow)
        assert bound.version != arena.version
        bound.carve(arena, split_module is None)
        assert bound.version == arena.version
        parts = arena.buffers["k2_stack"][: 3 * n].view(3, *split_src.shape)
        assert _geometry(bound.parts) == _geometry(parts)
        assert (bound.resid is None) == (split_module is not None)
        if plan.splits > 1:
            dims = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
            shape = (plan.splits, *dims)
            assert bound.partials_shape == shape
            buf = arena.buffers["k2_splitk_partials"]
            workspace = buf[: plan.splits * dims[0] * dims[1]].view(shape)
            assert _geometry(bound.workspace) == _geometry(workspace)
            out32, ldo, out_l = workspace, dims[1], dims[0] * dims[1]
            call_out = None
            # the partial sum of the partials into ``out`` (its flat alias when padded)
            assert entry.sum_entry is not None and bound.sum_template is not None
            sum_out = det_backend._k2_out32(out)
            assert bound.out_contiguous == (not case.endswith("padded_out"))
            sum_module = cake_jit.select_module(
                arch, cake_backend.PARTIALS_SUM_TEMPLATE
            )
            sum_grid = (-(-dims[0] * (-(-dims[1] // 4)) // 256), 1, 1)
            _same_arguments(
                bound.sum_arguments((a_view, sum_out, None, None)),
                cake_backend._order_atomic_arguments(
                    sum_module,
                    cake_jit.arg_plan_of(cake_jit.MODULES[sum_module]),
                    dict(
                        ws=workspace,
                        out=sum_out,
                        splits=plan.splits,
                        rows=dims[0],
                        cols=dims[1],
                        ldo=int(out.stride(0)),
                        enable_pdl=pdl,
                    ),
                    sum_grid,
                ),
            )
        else:
            assert bound.partials_shape is None and bound.workspace is None
            assert entry.sum_entry is None and bound.sum_template is None
            out32 = det_backend._k2_out32(out)
            assert bound.out_contiguous == (not case.endswith("padded_out"))
            ldo, out_l, call_out = int(out.stride(0)), 0, out32
        src = None if split_module is None else split_src
        kernel_args, split_args = bound.arguments((a_view, call_out, None, src))
        module_name = det_backend.select_module(arch, plan.template)
        _same_arguments(
            kernel_args,
            det_backend._order_arguments(
                module_name,
                det_backend.arg_plan_of(det_backend.MODULES[module_name]),
                det_backend._k2_values(
                    plan,
                    A=a_view,
                    parts=parts,
                    out32=out32,
                    ldo=ldo,
                    out_l=out_l,
                    enable_pdl=pdl,
                ),
                plan.grid,
            ),
        )
        if split_module is None:
            assert split_args is None
        else:
            _same_arguments(
                split_args,
                det_backend._order_arguments(
                    split_module,
                    det_backend.arg_plan_of(det_backend.MODULES[split_module]),
                    det_backend._split_values(split_src, parts, n, pdl),
                    det_backend._split_grid(n),
                ),
            )


def test_partials_sum_serves_two_to_eight_splits():
    """This package's partial sum serves the deterministic K2 program's partials ``[splits, rows, cols]`` with two to
    eight splits (the forward's four and the weight gradient's six among them) and two outputs or more, into an FP32
    ``out [rows, cols]`` with unit inner stride and a row stride of ``cols`` or more; every other launch keeps
    ``torch.sum``.  Its program is registered on every architecture of the package, one program for all."""
    assert list(cake_backend.PARTIALS_SUM_SPLITS) == [2, 3, 4, 5, 6, 7, 8]
    assert cake_backend.PARTIALS_SUM_TEMPLATE == "router_partials_sum"
    names = {
        cake_jit.select_module(arch, cake_backend.PARTIALS_SUM_TEMPLATE)
        for arch in cake_jit.ARCHES
    }
    assert len(names) == 1
    record = cake_jit.MODULES[names.pop()]
    assert record["arches"] == list(cake_jit.ARCHES)
    assert [name for _, name in cake_jit.arg_plan_of(record)] == [
        "ws",
        "out",
        "splits",
        "rows",
        "cols",
        "ldo",
        "enable_pdl",
        "grid_x",
        "grid_y",
        "grid_z",
    ]
    serves = cake_backend._partials_sum_serves
    out = torch.empty(64, 256)
    for splits in range(1, 11):
        assert serves((splits, 64, 256), out) == (2 <= splits <= 8), splits
    assert serves((6, 256, 6144), torch.empty(256, 6144))
    assert serves((4, 1, 8), torch.empty(1, 8))
    assert serves((4, 64, 256), torch.empty(64, 320)[:, :256])  # padded rows
    assert not serves((4, 1, 1), torch.empty(1, 1))  # one output: a full reduction
    assert not serves((4, 64, 256), torch.empty(256, 64).t())  # no unit inner stride
    assert not serves((4, 64, 256), torch.empty(64, 128))  # another shape
    assert not serves((4, 64, 256), torch.empty(64, 256, dtype=torch.float64))


def test_k1_view_recipes():
    """``_k1_view_recipe`` rebuilds a launch view from the call's tensor that holds its data pointer: the tensor as is,
    its ``unsqueeze(0)`` or one strided view of the same geometry (checked when bound); a pointer argument takes a
    contiguous tensor as is; a view of no call tensor, or of call tensors that alias, has no recipe."""
    SELF, UNSQ, STRIDED = (
        cake_backend._K1_VIEW_SELF,
        cake_backend._K1_VIEW_UNSQUEEZE,
        cake_backend._K1_VIEW_STRIDED,
    )
    X, W = (
        torch.zeros(64, 32, dtype=torch.bfloat16),
        torch.zeros(16, 32, dtype=torch.bfloat16),
    )
    out = torch.zeros(64, 16 + 8, dtype=torch.bfloat16)[:, :16]
    dense = torch.zeros(64, 16, dtype=torch.bfloat16)
    recipe = cake_backend._k1_view_recipe

    def rebuilt(view, sources, pointer=False):
        s, how, shape, strides = recipe(view, sources, pointer=pointer)
        t = sources[s]
        t = (
            t.unsqueeze(0)
            if how == UNSQ
            else t.as_strided(shape, strides)
            if how == STRIDED
            else t
        )
        assert t.data_ptr() == view.data_ptr() and t.dtype == view.dtype
        if not pointer:
            assert _geometry(t) == _geometry(view)
        return s, how

    assert rebuilt(X.unsqueeze(0), (X, W.t(), dense)) == (0, UNSQ)
    # the forward's B from W.t()
    assert rebuilt(W.unsqueeze(0), (X, W.t(), dense)) == (1, STRIDED)
    assert rebuilt(W.unsqueeze(0), (X, W, dense)) == (1, UNSQ)
    assert rebuilt(X.t().unsqueeze(0).transpose(1, 2), (X.t(), W, dense)) == (
        0,
        STRIDED,
    )
    batched = torch.zeros(4, 64, 32, dtype=torch.bfloat16)
    assert rebuilt(batched, (batched, W, dense)) == (0, SELF)
    flat = det_backend.flat_alias(dense.unsqueeze(0))
    # a contiguous out as the pointer
    assert rebuilt(flat, (X, W, dense), pointer=True) == (2, SELF)
    padded = det_backend.flat_alias(out.unsqueeze(0))
    assert rebuilt(padded, (X, W, out), pointer=True) == (2, STRIDED)
    assert rebuilt(out.unsqueeze(0), (X, W, out)) == (2, UNSQ)
    # another tensor's data, another dtype
    assert (
        recipe(
            torch.zeros(1, 64, 32, dtype=torch.bfloat16), (X, W, dense), pointer=False
        )
        is None
    )
    assert (
        recipe(X.view(torch.int16).unsqueeze(0), (X, W, dense), pointer=False) is None
    )
    # call tensors that start at the same data pointer (aliasing is not part of the key): no recipe
    assert recipe(X.unsqueeze(0), (X, X[:, :16], dense), pointer=False) is None
    assert recipe(flat, (X, dense.view(64, 16), dense), pointer=True) is None


def _cpu_k1_launches(monkeypatch, arch="sm_103a"):
    """CPU binding tests of the K1 launches (nothing is loaded or launched): the device facts of a 152-SM part, the
    deterministic package's dummies on the CPU, one stand-in module whose entry records its arguments, the FFI
    conversion as the identity; returns that entry.  The one-split ranges of :data:`K1_ONE_SPLIT_RANGES_BY_SM` and
    the cluster ranges of :data:`K1_DSM_RANGES_BY_SM` are taken out (on 152 SMs they hold rows these tests call as
    rows without a verdict; ``test_k1_152_sm_one_split_ranges_bind_the_one_split_program`` and
    ``test_k1_152_sm_dsm_ranges_bind_the_cluster_program`` bind them)."""
    monkeypatch.setattr(cake_backend, "K1_ONE_SPLIT_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend.tvm_ffi, "from_dlpack", lambda t, **_: t)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=152, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "_same_cuda_device", lambda A, B, out: out.device)
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(det_backend, "require_arch", lambda device: arch)
    monkeypatch.setattr(det_backend, "_device_index", lambda device: 0)
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 0)
    monkeypatch.setattr(
        det_backend, "_dummy_key", lambda device, dtype, tag=None: ("cpu", dtype, tag)
    )
    monkeypatch.setattr(det_backend, "_DUMMIES", {})
    monkeypatch.setattr(
        det_backend,
        "_ffi_stream_context",
        lambda index, raw=None: contextlib.nullcontext(),
    )
    module = _FakeModule()
    monkeypatch.setattr(
        det_backend,
        "load_cake_dense_projection_det_gemm_module",
        lambda name, a: module,
    )
    monkeypatch.setattr(
        cake_backend, "load_cake_dense_projection_gemm_module", lambda name, a: module
    )
    monkeypatch.setattr(cake_backend, "_ATOMIC_ARENAS", {})
    monkeypatch.setattr(cake_backend, "_K1_STREAM_CONTEXTS", {})
    for cache in ("_K1_ATOMIC_CACHE", "_K1_DET_CACHE", "_K1_PROGRAM_CACHE"):
        monkeypatch.setattr(cake_backend, cache, type(getattr(cake_backend, cache))())
    return module.entry


def _same_k1_arguments(got: list, want: list, arg_plan: list) -> None:
    """A bound K1 launch's arguments are the per-call ordered ones: the same view geometry for a TMA operand, the
    same data pointer and dtype (and a contiguous tensor) for a pointer argument, the same values otherwise."""
    kinds = [kind for kind, _ in arg_plan]
    assert len(got) == len(want) == len(kinds)
    for i, (g, w, kind) in enumerate(zip(got, want, kinds, strict=True)):
        if isinstance(w, torch.Tensor):
            assert isinstance(g, torch.Tensor), i
            if kind == "buffer":
                assert (g.data_ptr(), g.dtype) == (w.data_ptr(), w.dtype), i
                assert g.is_contiguous(), i
            else:
                assert _geometry(g) == _geometry(w), i
        else:
            assert type(g) is type(w) and g == w, (i, g, w)


def _k1_det_call(case, arch):
    """``(sources, (A, B, out, transposed_out, knobs))`` of a K1 call of the deterministic program on fresh CPU
    tensors: the entry point's own tensors and the deterministic package's convenience call."""
    bf16 = torch.bfloat16
    T, K, N = (64 if case == "fwd_few_tokens" else 1000), 4096, 128
    X, W, G = (
        torch.empty(T, K, dtype=bf16),
        torch.empty(N, K, dtype=bf16),
        torch.empty(T, N, dtype=bf16),
    )
    if case.startswith("fwd"):
        out = (
            torch.empty(T, N + 64, dtype=bf16)[:, :N]
            if case.endswith("padded_out")
            else torch.empty(T, N, dtype=bf16)
        )
        return (X, W, out), (X, W.t(), out, False, ())
    if case.startswith("dgrad"):
        out = torch.empty(T, K, dtype=torch.float32 if case.endswith("f32") else bf16)
        return (G, W, out), (G, W, out, False, ())
    if case.startswith("wgrad"):
        if case.endswith("wide"):
            G = torch.empty(T, 2048, dtype=bf16)
        out = torch.empty(G.shape[1], K, dtype=torch.float32 if "f32" in case else bf16)
        A, B, transposed = det_backend.wgrad_views(
            G, X, arch=arch, out_f32=out.dtype == torch.float32
        )
        knobs = (("cta_rows", None), ("sk", False if case.endswith("sk") else "auto"))
        return (X.t(), G, out), (A, B, out, transposed, knobs)
    # the batched MLA forward: strided head views of [T, H, slot] activations
    H, slot, D_in, D_out = 8, 256, 192, 512
    act = torch.empty(T, H, slot, dtype=bf16)[..., :D_in]
    A, B = act.permute(1, 0, 2), torch.empty(H, D_in, D_out, dtype=bf16)
    out = torch.empty(T, H, D_out, dtype=bf16).permute(1, 0, 2)
    return (A, B, out), (A, B, out, False, ())


K1_DET_CASES = [
    "fwd", "fwd_padded_out", "fwd_few_tokens", "dgrad", "dgrad_f32", "wgrad_f32", "wgrad", "wgrad_f32_wide",
    "wgrad_f32_sk", "batched",
]  # fmt: skip


@pytest.mark.parametrize("pdl", [0, 1])
@pytest.mark.parametrize("case", K1_DET_CASES)
def test_k1_det_binding_is_the_deterministic_launch(monkeypatch, case, pdl):
    """The binding of the deterministic K1 program on this package's convenience path (``_plan_det_k1``) is the
    deterministic package's plan and program with its argument list ordered once: with the views of a later call's
    own tensors (new data pointers) in place, the arguments are the ones the deterministic package's convenience path
    resolves for that call (``_k1_views`` / ``_k1_values``; ``out`` itself for its pointer when contiguous), with the
    stream-K slabs and slice counters of this package's arena -- carved again after the arena grows -- and the bound
    PDL flag."""
    arch = "sm_103a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    sources, (A, B, out, transposed, knobs) = _k1_det_call(case, arch)
    entry = cake_backend._plan_det_k1(A, B, out, transposed, knobs, sources, 0, 0, pdl)
    plan, *_, module_name = det_backend._k1_plan(
        A, B, out, transposed_out=transposed, knobs=dict(knobs)
    )
    assert entry.plan == plan and entry.k1 is not None and entry.entry is entry_fn
    arena = entry.arena
    assert arena.buffers["k1_det_counters"].dtype == torch.uint32
    arg_plan = det_backend.arg_plan_of(det_backend.MODULES[module_name])
    for grow in (1, 4):
        if grow > 1:
            arena.require("k1_det_ws", arena.buffers["k1_det_ws"].numel() * grow)
        # a later call: fresh tensors of the key's geometry
        sources2, (A2, B2, out2, _, _) = _k1_det_call(case, arch)
        entry_fn.calls.clear()
        cake_backend._launch_k1_binding(
            det_backend, entry.entry, entry.k1, arena, sources2, 0, 0
        )
        assert entry.k1.version == arena.version
        ((got, _),) = entry_fn.calls
        swapped = det_backend._k1_swapped(plan, A2, B2, transposed_out=transposed)
        a_desc, b_desc, O3 = det_backend._k1_views(A2, B2, out2, swapped=swapped)
        ldo, out_l = det_backend._out_strides(O3)
        values = det_backend._k1_values(
            plan,
            out2.device,
            A=a_desc,
            B=b_desc,
            out3=O3,
            o_alias=det_backend.flat_alias(O3),
            ws=arena.buffers["k1_det_ws"],
            counters=arena.buffers["k1_det_counters"],
            ldo=ldo,
            out_l=out_l,
            enable_pdl=pdl,
        )
        want = det_backend._order_arguments(module_name, arg_plan, values, plan.grid)
        _same_k1_arguments(got, want, arg_plan)
        assert got[[name for _, name in arg_plan].index("enable_pdl")] == pdl


@pytest.mark.parametrize("pdl", [0, 1])
@pytest.mark.parametrize("case", ["two_splits", "one_split", "padded_out"])
def test_k1_atomic_binding_is_the_ordered_launch(monkeypatch, case, pdl):
    """A cached K1 launch of this package binds its argument list once (``_bind_atomic_k1``): with a later call's
    views in place it is the list the unbound launch orders per call (``_launch_atomic_k1_unbound``), with the
    arena's split slots and tickets (several splits) or the dummies (one split), and the PDL flag."""
    arch = "sm_103a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    splits = 1 if case == "one_split" else 2
    T, N, K = 1000, 32, 4096

    def call():
        G, X = (
            torch.empty(T, N, dtype=torch.bfloat16),
            torch.empty(T, K, dtype=torch.bfloat16),
        )
        out = (
            torch.empty(N, K + 64)[:, :K]
            if case == "padded_out"
            else torch.empty(N, K, dtype=torch.float32)
        )
        return X.t(), G, out

    A, B, out = call()
    plan, *_ = cake_backend.plan_atomic_gemm(
        A, B, out, transposed_out=True, arch=arch, sm_count=152, splits=splits
    )
    assert plan is not None and plan.splits == splits
    module_name = cake_jit.select_module(arch, plan.template)
    fn, arg_plan = cake_backend._atomic_entry(module_name, plan.grid, arch)
    arena = None
    if splits > 1:
        arena = cake_backend._atomic_arena_for(0, 0, out.device)
        arena.require("k1_slots", plan.ws_f32_elems)
        arena.require("k1_tickets", plan.counters_u32)
    bound = cake_backend._bind_atomic_k1(
        plan, module_name, arg_plan, A, B, out, arena, pdl
    )
    assert bound is not None and (bound.workspace == ()) == (arena is None)
    entry = cake_backend._CachedAtomic(
        plan=plan,
        arch=arch,
        module_name=module_name,
        entry=fn,
        arg_plan=arg_plan,
        arena=arena,
    )
    A2, B2, out2 = call()
    cake_backend._launch_k1_binding(det_backend, fn, bound, arena, (A2, B2, out2), 0, 0)
    cake_backend._launch_atomic_k1_unbound(entry, A2, B2, out2, 0, 0, pdl)
    (got, _), (want, _) = entry_fn.calls
    _same_k1_arguments(got, want, arg_plan)
    assert got[[name for _, name in arg_plan].index("enable_pdl")] == pdl


def test_k1_convenience_calls_cache_their_bindings(monkeypatch):
    """The K1 entry points plan once per key and launch the bound program after: the
    forward and the input gradient in ``_K1_DET_CACHE`` (keyed by their own tensors), the weight gradient in
    ``_K1_ATOMIC_CACHE`` -- a row without a verdict bound to the deterministic package's plan of
    ``cake_dense_gemm_det_train_projection_wgrad`` (its orientation; another ``sk`` is another key and plan), a row with one to this
    package's program."""
    arch = "sm_103a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16, T = torch.bfloat16, 1000
    X, W, G = (
        torch.empty(T, 4096, dtype=bf16),
        torch.empty(128, 4096, dtype=bf16),
        torch.empty(T, 128, dtype=bf16),
    )
    out_f, out_d, out_w = (
        torch.empty(T, 128, dtype=bf16),
        torch.empty(T, 4096, dtype=bf16),
        torch.empty(128, 4096),
    )
    for _ in range(2):
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, out_f)
            is out_f
        )
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out_d)
            is out_d
        )
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w)
            is out_w
        )
    assert len(entry_fn.calls) == 6
    det_entries = list(cake_backend._K1_DET_CACHE.items())
    # the flags, the PDL flag (the device's default: on) and ``weights_static``
    assert [key[2] for key, _ in det_entries] == [
        (("fwd",), 1, False),
        (("dgrad",), 1, False),
    ]
    assert all(e.k1 is not None and e.arena is not None for _, e in det_entries)
    ((key, wgrad),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert wgrad.plan is None and wgrad.k1 is not None and key[2][3] == ("auto", None)
    assert key[2][4:] == (1, False)
    # the PDL flag is bound into the argument list: another flag is another key
    entry_fn.calls.clear()
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_forward(
            X, W, out_f, enable_pdl=False
        )
        is out_f
    )
    assert next(reversed(cake_backend._K1_DET_CACHE))[2] == (("fwd",), 0, False)
    module_name = det_backend._k1_plan(X, W.t(), out_f, transposed_out=False, knobs={})[
        -1
    ]
    names = [n for _, n in det_backend.arg_plan_of(det_backend.MODULES[module_name])]
    ((args, _),) = entry_fn.calls
    assert args[names.index("enable_pdl")] == 0
    # the deterministic package's orientation of the call (its architecture lookup of ``out``'s device)
    A, B, transposed = det_backend.wgrad_views(
        G, X, arch=det_backend.arch_for(out_w.device), out_f32=True
    )
    assert (
        wgrad.det_plan
        == det_backend._k1_plan(
            A, B, out_w, transposed_out=transposed, knobs=dict(sk="auto", cta_rows=None)
        )[0]
    )
    cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w, sk=False)
    assert len(cake_backend._K1_ATOMIC_CACHE) == 2
    sk_entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
    assert (
        sk_entry.det_plan
        == det_backend._k1_plan(
            A, B, out_w, transposed_out=transposed, knobs=dict(sk=False, cta_rows=None)
        )[0]
    )
    # a row with a verdict: this package's program, bound
    key = (arch, 1, 4096, 128, "f32")
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (128, 2))
    monkeypatch.delitem(cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS, key, raising=False)
    cake_backend._K1_ATOMIC_CACHE.clear()
    cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w)
    ((_, atomic),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert atomic.plan is not None and atomic.plan.splits == 2 and atomic.k1 is not None
    assert atomic.det_plan is None and atomic.arena is not None


#: The L2 size of the 152-SM GB200 and GB300 parts (``L2_cache_size``) the forward rows were measured on.
GB_L2_BYTES = 135528448
#: The rows of :data:`K1_FORWARD_RULES_BY_SM`, ``(arch, sm_count) -> {(K, N, T), ...}``: the forward of exactly these
#: ``K -> N`` rows and token counts runs the single-CTA program on 152-SM sm_100a and sm_103a parts.
K1_FORWARD_ROWS = {
    ("sm_100a", 152): {(4096, 512, 2048), (5120, 512, 1024), (5120, 512, 4096)},
    ("sm_103a", 152): {(4096, 512, 2048), (5120, 512, 1024)},
}  # fmt: skip
#: The forward rows measured on those parts that missed and keep the deterministic plan (keys as in K1_FORWARD_ROWS).
K1_FORWARD_MISSED_ROWS = {
    ("sm_100a", 152): {(4096, 512, 3072), (4096, 512, 4096)},
    ("sm_103a", 152): {(4096, 512, 3072), (4096, 512, 4096), (5120, 512, 4096)},
}  # fmt: skip
#: The ``K -> N`` rows of the forward next to those of the table (and theirs) that the exact keys must leave alone.
K1_FORWARD_NEAR_ROWS = ((4096, 512), (5120, 512), (4096, 128), (5120, 128), (4096, 1024), (6144, 512))  # fmt: skip


def test_k1_forward_rules_by_sm():
    """:data:`K1_FORWARD_RULES_BY_SM` holds the measured forward rows only, each with the single-CTA knobs, which the
    deterministic planner turns into its single-CTA program (128 x 128 tiles over the whole K, no stream-K units) where
    its own plan of the row runs CTA pairs of 256-wide tiles with stream-K units.  :func:`k1_forward_knobs` keys
    exactly: every other architecture, SM count, row and token count (the measured rows that missed among them:
    ``4096 -> 512`` at 3072 and 4096 tokens, ``5120 -> 512`` at 4096 tokens on ``sm_103a``) gets no knobs, i.e. the
    deterministic plan."""
    full = cake_backend.K1_FORWARD_RULES_BY_SM
    # the knob rows (the rows naming a program: test_k1_forward_named_rows_plan_their_program)
    table = {
        part: {key: v for key, v in rows.items() if not isinstance(v, str)}
        for part, rows in full.items()
    }
    table = {part: rows for part, rows in table.items() if rows}
    single = cake_backend.K1_FORWARD_SINGLE_CTA_KNOBS
    assert single == {"cta1": True, "block_n": 128, "hints": ("none", "none")}
    assert {part: set(rows) for part, rows in table.items()} == K1_FORWARD_ROWS
    assert all(knobs == single for rows in table.values() for knobs in rows.values())
    bf16 = torch.bfloat16
    for (arch, sms), rows in table.items():
        for K, N, T in rows:
            kw = dict(arch=arch, sm_count=sms, K=K, N=N, T=T)
            knobs = cake_backend.k1_forward_knobs(**kw)
            assert knobs == tuple(sorted(single.items())), kw
            X, W = torch.empty(T, K, dtype=bf16), torch.empty(N, K, dtype=bf16)
            out = torch.empty(T, N, dtype=bf16)

            def plan(**kw):
                return det_backend.cake_dense_gemm_det_train_projection_plan(
                    X, W.t(), out, sm_count=sms, l2_bytes=GB_L2_BYTES, arch=arch, **kw
                )[0]

            own, default = plan(**dict(knobs)), plan()
            assert own.template.endswith("_kk_n128_c1"), (arch, K, N, T, own.template)
            assert int(own.sk_units) == 0, (arch, K, N, T)
            assert tuple(own.grid) == ((T // 128) * (N // 128), 1, 1), (arch, K, N, T)
            assert default.template.endswith("_kk_n256_pro7"), (arch, K, N, T)
            assert int(default.sk_units) > 0, (arch, K, N, T)
    near = (-64, -1, 0, 1, 64)
    parts = (*K1_FORWARD_ROWS.values(), *K1_FORWARD_MISSED_ROWS.values())
    Ts = {T + d for rows in parts for _, _, T in rows for d in near}
    Ts |= {512, 4096, 8192, 16384}
    routed = 0
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for sms in (148, 150, 151, 152, 153, 160, 208, 212, 216):
            for K, N in K1_FORWARD_NEAR_ROWS:
                for T in sorted(Ts):
                    kw = dict(arch=arch, sm_count=sms, K=K, N=N, T=T)
                    got = cake_backend.k1_forward_knobs(**kw)
                    held = (K, N, T) in K1_FORWARD_ROWS.get((arch, sms), set())
                    named = (K, N, T) in full.get((arch, sms), {}) and not held
                    assert bool(got) == (held or named), (arch, sms, K, N, T)
                    routed += held
    assert routed == sum(len(v) for v in K1_FORWARD_ROWS.values())
    for (arch, sms), rows in K1_FORWARD_MISSED_ROWS.items():
        assert not rows & K1_FORWARD_ROWS[(arch, sms)], arch
        for K, N, T in rows:
            kw = dict(arch=arch, sm_count=sms, K=K, N=N, T=T)
            assert cake_backend.k1_forward_knobs(**kw) == (), kw


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_k1_forward_rows_bind_their_plan(monkeypatch, arch):
    """``cake_dense_gemm_nondet_train_projection_forward()`` binds, for a row of :data:`K1_FORWARD_RULES_BY_SM` on its part
    (BF16 ``out``, both operands K-major), the deterministic program the deterministic planner gives the row with the
    table's knobs, and the deterministic plan for a token count, row or SM count outside the table; an FP32 ``out``
    or an operand that is not K-major gets no knobs."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    single = tuple(sorted(cake_backend.K1_FORWARD_SINGLE_CTA_KNOBS.items()))

    def bound_plan(X, W, out):
        cake_backend._K1_DET_CACHE.clear()
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, out)
            is out
        )
        ((key, entry),) = cake_backend._K1_DET_CACHE.items()
        assert key[2] == (("fwd",), 1, False) and entry.k1 is not None
        return entry.plan

    def det_plan(X, W, out, knobs):
        kw = dict(transposed_out=False, knobs=dict(knobs))
        return det_backend._k1_plan(X, W.t(), out, **kw)[0]

    rows = sorted(K1_FORWARD_ROWS[(arch, 152)])
    others = sorted(K1_FORWARD_MISSED_ROWS[(arch, 152)])
    others += [(4096, 512, 2049), (4096, 512, 2047), (5120, 512, 1023), (4096, 128, 2048), (5120, 1024, 1024)]  # fmt: skip
    for K, N, T in rows + others:
        X, W = torch.empty(T, K, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out, what = torch.empty(T, N, dtype=bf16), (K, N, T)
        knobs = single if (K, N, T) in rows else ()
        assert cake_backend._k1_forward_knobs_of(X, W, out) == knobs, what
        plan = bound_plan(X, W, out)
        assert plan == det_plan(X, W, out, knobs), what
        assert plan.template.endswith("_kk_n128_c1") == bool(knobs), what
        f32 = torch.empty(T, N, dtype=torch.float32)
        assert cake_backend._k1_forward_knobs_of(X, W, f32) == (), what
    assert len(entry_fn.calls) == len(rows + others)
    # an operand that is not K-major (the same shapes): no knobs, the deterministic plan
    K, N, T = rows[0]
    out = torch.empty(T, N, dtype=bf16)
    Xmn = torch.empty(K, T, dtype=bf16).t()
    Wmn = torch.empty(K, N, dtype=bf16).t()
    X, W = torch.empty(T, K, dtype=bf16), torch.empty(N, K, dtype=bf16)
    assert cake_backend._k1_forward_knobs_of(Xmn, W, out) == ()
    assert cake_backend._k1_forward_knobs_of(X, Wmn, out) == ()
    # another SM count of the architecture: the deterministic plan
    for sms in (148, 151, 153, 160):
        facts = dataclasses.replace(det_backend.device_facts(), sm_count=sms)
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        assert cake_backend._k1_forward_knobs_of(X, W, out) == (), sms
        assert bound_plan(X, W, out) == det_plan(X, W, out, ()), sms


#: The rows of the program tables, ``(arch, sm_count) -> {key: template}`` (keys as in the tables): the lean
#: one-wave forward of eight epilogue warps with the X prefetch and the token-half last stage, at 512 features on
#: every part of the table and at 1024 features on the 152-SM parts; the 512-feature rows at 16172 and 16231 tokens
#: with the L2 eviction priorities (``_lh``).
_WAVE_PF2_ST = "dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st"
_WAVE_PF2_ST_LH = "dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st_lh"
_F512_ROWS = {(K, 512, T) for K in (4096, 5120) for T in (16172, 16231, 16384)}
_F1024_ROWS = {(K, 1024, T) for K in (4096, 5120) for T in (16172, 16231, 16384)}
_F512_LH = dict.fromkeys(
    ((K, 512, T) for K in (4096, 5120) for T in (16172, 16231)), _WAVE_PF2_ST_LH
)
K1_FORWARD_PROGRAM_ROWS = {
    ("sm_100a", 148): {**dict.fromkeys(_F512_ROWS, _WAVE_PF2_ST), **_F512_LH},
    ("sm_100a", 152): {
        **dict.fromkeys(_F512_ROWS | _F1024_ROWS, _WAVE_PF2_ST),
        **_F512_LH,
    },
    ("sm_103a", 152): {
        **dict.fromkeys(_F512_ROWS | _F1024_ROWS, _WAVE_PF2_ST),
        **_F512_LH,
    },
}
#: The rows of :data:`K1_DGRAD_NVJET_RULES_BY_SM` (BF16 out, all on the lean nvjet program): ``4096 -> 32`` and
#: ``4096 -> 64`` at the token counts of the thin scan, ``4096 -> 16, 48, 96, 128``, ``5120 -> 32, 48, 64, 128``,
#: ``6144 -> 32, 128`` and ``8192 -> 128`` at 16231 tokens; sm_103a leaves ``4096 -> 32`` at 2048 tokens and sm_100a
#: three rows to the other tables.
_NVJET = "dense_proj_gemm_nvjet_early_kn_n256_lean"
_NVJET_ROWS = {
    *(
        (4096, N, T)
        for N in (32, 64)
        for T in (512, 2048, 4864, 9472, 9728, 14592, 16172, 16231, 16384)
    ),
    *((4096, N, 16231) for N in (16, 48, 96, 128)),
    *((5120, N, 16231) for N in (32, 48, 64, 128)),
    (6144, 32, 16231),
    (6144, 128, 16231),
    (8192, 128, 16231),
}
_NVJET_SM103_KEPT = {(4096, 32, 2048)}
_NVJET_SM100_KEPT = _NVJET_SM103_KEPT | {(4096, 32, 14592), (4096, 64, 4864)}
#: The rows of :data:`K1_DGRAD_NVJET_RULES_BY_SM` on the program that frees the tensor memory under the last store
#: drain (``_lean_free``) per part.
_NVJET_FREE = "dense_proj_gemm_nvjet_early_kn_n256_lean_free"
_GLM = (16172, 16231, 16384)
_NVJET_FREE_ROWS = {
    ("sm_100a", 148): {
        *(
            (K, N, T)
            for K, N in (
                (4096, 32),
                (4096, 48),
                (4096, 128),
                (5120, 32),
                (6144, 32),
                (6144, 128),
            )
            for T in _GLM
        ),
        *(
            (K, N, T)
            for K, N in ((5120, 48), (5120, 64), (5120, 128), (8192, 128))
            for T in (16172, 16384)
        ),
    },
    ("sm_103a", 152): {
        *(
            (K, N, T)
            for K, N in ((4096, 128), (5120, 128), (6144, 32), (6144, 128))
            for T in _GLM
        ),
        *(
            (K, N, T)
            for K, N in ((4096, 48), (5120, 32), (5120, 48), (5120, 64), (8192, 128))
            for T in (16172, 16384)
        ),
    },
    ("sm_107a", 212): {
        (K, N, T) for K in (4096, 5120) for N in (32, 48, 64, 128) for T in _GLM
    }
    | {(K, N, T) for K, N in ((6144, 32), (6144, 128), (8192, 128)) for T in _GLM},
}
K1_DGRAD_NVJET_ROWS = {
    ("sm_100a", 148): {
        **{(*r, "bf16"): _NVJET for r in _NVJET_ROWS - _NVJET_SM100_KEPT},
        **{(*r, "bf16"): _NVJET_FREE for r in _NVJET_FREE_ROWS[("sm_100a", 148)]},
    },
    ("sm_100a", 152): {(*r, "bf16"): _NVJET for r in _NVJET_ROWS - _NVJET_SM100_KEPT},
    ("sm_103a", 152): {
        **{(*r, "bf16"): _NVJET for r in _NVJET_ROWS - _NVJET_SM103_KEPT},
        **{(*r, "bf16"): _NVJET_FREE for r in _NVJET_FREE_ROWS[("sm_103a", 152)]},
    },
    ("sm_107a", 212): {
        (*r, "bf16"): _NVJET_FREE for r in _NVJET_FREE_ROWS[("sm_107a", 212)]
    },
}
_KD, _KD_TMA, _KD_TMA_LEAN_FREE = (
    "dense_proj_gemm_dgrad_kn_n256_f32_ef",
    "dense_proj_gemm_dgrad_kn_n256_f32_ef_tma",
    "dense_proj_gemm_dgrad_kn_n256_f32_tma_lean_free",
)
#: The token window of the 148-SM ``sm_100a`` and 152-SM ``sm_103a`` input-gradient program rows: 16172 to 16384
#: tokens; the GPU tests run its rows at its ends, the token counts measured inside it and one more.
_KD_WINDOW = range(16172, 16385)
_KD_WINDOW_SAMPLE = frozenset({16172, 16231, 16300, 16384})
K1_DGRAD_PROGRAM_ROWS = {
    ("sm_100a", 148): {
        **{
            (K, 1024, T, "f32"): _KD
            for K in (4096, 5120)
            for T in (16172, 16231, 16384)
        },
        **{
            (K, N, T, "f32"): _KD_TMA_LEAN_FREE
            for K, N in ((4096, 512), (5120, 512), (6144, 576))
            for T in _KD_WINDOW
        },
    },
    ("sm_100a", 152): {
        (K, N, T, "f32"): _KD_TMA
        for K in (4096, 5120)
        for N in (512, 1024)
        for T in (16172, 16231, 16384)
    },
    ("sm_103a", 152): {
        **{
            (K, 1024, T, "f32"): _KD
            for K in (4096, 5120)
            for T in (16172, 16231, 16384)
        },
        **{
            (K, N, T, "f32"): _KD_TMA_LEAN_FREE
            for K, N in ((4096, 512), (5120, 512), (6144, 576))
            for T in _KD_WINDOW
        },
    },
}
K1_WGRAD_SK_ROWS = {
    ("sm_100a", 148): {
        (5120, 1024, T, dt): "dense_proj_gemm_sk_nn_n256"
        + ("_f32" if dt == "f32" else "")
        + "_t"
        for T in (16172, 16231, 16384)
        for dt in ("bf16", "f32")
    },
    ("sm_103a", 152): {
        (5120, 1024, 16172, "bf16"): "dense_proj_gemm_sk_nn_n256_t",
        (5120, 1024, 16231, "f32"): "dense_proj_gemm_sk_nn_n256_f32_t",
        (5120, 1024, 16384, "f32"): "dense_proj_gemm_sk_nn_n256_f32_t",
        (5120, 1024, 16100, "f32"): "dense_proj_gemm_sk_nn_n256_f32_t",
        (5120, 1024, 16172, "f32"): "dense_proj_gemm_sk_nn_n256_f32_t",
        (5120, 1024, 16300, "f32"): "dense_proj_gemm_sk_nn_n256_f32_t",
    },
    ("sm_107a", 212): {
        (4096, 1024, 16172, "bf16"): "dense_proj_gemm_sk_nn_n256_t",
        (4096, 1024, 16384, "bf16"): "dense_proj_gemm_sk_nn_n256_t",
    },
}
#: The 64-token K-block counts of 16172 to 16384 tokens: 253 (odd), 254, 255 (odd) and 256.
_KB_PARITY_RANGES = (
    (16172, 16192, 4),
    (16193, 16256, 2),
    (16257, 16320, 4),
    (16321, 16384, 2),
)
K1_ORDERED_ROWS = {
    ("sm_100a", 148): {
        ("sm_100a", 1, 4096, 1024, "bf16"): ((961, 16384, (256, 1, 2, "arrival")),),
    },
    ("sm_107a", 212): {
        ("sm_107a", 1, 4096, 32, "f32"): ((16172, 16384, (64, 4, 1, "mirror")),),
        **{
            ("sm_107a", 1, d_in, 1024, "f32"): tuple(
                (lo, hi, (bn, 1, chains, "arrival"))
                for lo, hi, chains in _KB_PARITY_RANGES
            )
            for d_in, bn in ((4096, 192), (5120, 256))
        },
        ("sm_107a", 1, 5120, 512, "f32"): tuple(
            (lo, hi, (128, 1, 2, "arrival"))
            for lo, hi, chains in _KB_PARITY_RANGES
            if chains == 2
        ),
    },
}


def test_k1_program_tables():
    """The forward, input-gradient (this package's programs, and the nvjet program on the deterministic plan of
    :data:`K1_DGRAD_NVJET_DET_PROGRAM`) and stream-K weight-gradient program tables hold exactly the measured rows,
    each naming a template this checkout registers for the table's architecture; the ranges of
    :data:`K1_ORDERED_RANGES_BY_SM` hold the measured rows and plan their program (:func:`k1_ordered_plan`) at their
    token counts only, on their part only, before :func:`k1_plan`'s routing."""
    assert len(_NVJET_ROWS) == 29
    assert {part: len(rows) for part, rows in _NVJET_FREE_ROWS.items()} == {
        ("sm_100a", 148): 26,
        ("sm_103a", 152): 22,
        ("sm_107a", 212): 33,
    }
    for table, want in (
        (cake_backend.K1_FORWARD_PROGRAM_RULES_BY_SM, K1_FORWARD_PROGRAM_ROWS),
        (cake_backend.K1_DGRAD_PROGRAM_RULES_BY_SM, K1_DGRAD_PROGRAM_ROWS),
        (cake_backend.K1_WGRAD_SK_RULES_BY_SM, K1_WGRAD_SK_ROWS),
        (cake_backend.K1_DGRAD_NVJET_RULES_BY_SM, K1_DGRAD_NVJET_ROWS),
    ):
        assert table == want
        for (arch, _), rows in table.items():
            for template in rows.values():
                assert template in cake_jit.KERNELS[arch], (arch, template)
    assert cake_backend.dgrad_program_template(False) == _KD
    assert cake_backend.dgrad_program_template(True) == _KD_TMA
    assert (
        cake_backend.dgrad_program_template(True, lean_free=True) == _KD_TMA_LEAN_FREE
    )
    assert (
        cake_backend.wgrad_sk_template(256, True) == "dense_proj_gemm_sk_nn_n256_f32_t"
    )
    assert cake_backend.K1_ORDERED_RANGES_BY_SM == K1_ORDERED_ROWS
    assert (
        cake_backend.atomic_k1_template(64, True, 1, "mirror")
        == "dense_proj_gemm_atomic_nn_n64_f32_t_mirror"
    )
    assert (
        cake_backend.atomic_k1_template(192, True, 2)
        == "dense_proj_gemm_atomic_nn_n192_f32_c2"
    )
    with pytest.raises(ValueError, match="split_order"):
        cake_backend.atomic_k1_template(64, True, 1, "index")
    for (arch, sms), rows in K1_ORDERED_ROWS.items():
        for key, ranges in rows.items():
            _, L, M, N, dt = key
            kw = dict(arch=arch, L=L, M=M, N=N, out_f32=dt == "f32", sm_count=sms)
            for lo, hi, value in ranges:
                bn, splits, chains, order = value
                assert (
                    cake_backend.atomic_k1_template(bn, dt == "f32", chains, order)
                    in cake_jit.KERNELS[arch]
                )
                for T in sorted({lo, hi, (lo + hi) // 2}):
                    assert cake_backend.k1_ordered_plan(K=T, **kw) == value, (key, T)
                for T in (lo - 1, hi + 1):
                    # next to a range: the row's adjacent range (the K-block parity ranges), else no range
                    near = next((v for a, b, v in ranges if a <= T <= b), None)
                    assert cake_backend.k1_ordered_plan(K=T, **kw) == near, (key, T)
                for other in (sms - 4, sms - 1, sms + 1, sms + 4):
                    assert (
                        cake_backend.k1_ordered_plan(K=lo, **dict(kw, sm_count=other))
                        is None
                    )
                assert (
                    cake_backend.k1_ordered_plan(K=lo, **dict(kw, out_f32=dt != "f32"))
                    is None
                )
                assert cake_backend.k1_ordered_plan(K=lo, **dict(kw, L=2)) is None


def test_k1_ordered_ranges_plan_their_program(monkeypatch):
    """A token count of :data:`K1_ORDERED_RANGES_BY_SM` plans its program (``split_order``, the template's suffix, the
    split and chain counts; the split workspace with more than one split) in place of the split tables' choice, a plan
    that repeats (four splits in the mirror order, or one split of two or four chains, by the parity of the K-block
    count on 212-SM sm_107a; two chains at either parity on 148-SM sm_100a); one token outside the ranges keeps that
    choice.  :data:`K1_PARITY_RANGES_BY_SM`, which comes first and holds these 212-SM token counts too, is taken out
    (``test_k1_parity_ranges_plan_their_program`` holds it)."""
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    arch = "sm_107a"
    _register_k1(monkeypatch, arch)
    kw = dict(transposed_out=True, arch=arch, sm_count=212)

    def operands(T, d_in, d_out):
        G = torch.zeros(T, d_out, dtype=torch.bfloat16)
        X = torch.zeros(T, d_in, dtype=torch.bfloat16)
        return X.t(), G, torch.zeros(d_out, d_in, dtype=torch.float32)

    for d_in, d_out, T, template, splits, chains, order in (
        (4096, 32, 16172, "dense_proj_gemm_atomic_nn_n64_f32_t_mirror", 4, 1, "mirror"),
        (4096, 32, 16231, "dense_proj_gemm_atomic_nn_n64_f32_t_mirror", 4, 1, "mirror"),
        (4096, 32, 16384, "dense_proj_gemm_atomic_nn_n64_f32_t_mirror", 4, 1, "mirror"),
        (4096, 1024, 16384, "dense_proj_gemm_atomic_nn_n192_f32_c2", 1, 2, "arrival"),
        # the K-block parity of 16172 to 16384 tokens: four chains at an odd count, two at an even count
        (4096, 1024, 16172, "dense_proj_gemm_atomic_nn_n192_f32_c4", 1, 4, "arrival"),
        (4096, 1024, 16300, "dense_proj_gemm_atomic_nn_n192_f32_c4", 1, 4, "arrival"),
        (4096, 1024, 16231, "dense_proj_gemm_atomic_nn_n192_f32_c2", 1, 2, "arrival"),
        (5120, 1024, 16172, "dense_proj_gemm_atomic_nn_n256_f32_t_c4", 1, 4, "arrival"),
        (5120, 1024, 16231, "dense_proj_gemm_atomic_nn_n256_f32_t_c2", 1, 2, "arrival"),
        (5120, 1024, 16384, "dense_proj_gemm_atomic_nn_n256_f32_t_c2", 1, 2, "arrival"),
        (5120, 512, 16231, "dense_proj_gemm_atomic_nn_n128_f32_t_c2", 1, 2, "arrival"),
        (5120, 512, 16384, "dense_proj_gemm_atomic_nn_n128_f32_t_c2", 1, 2, "arrival"),
    ):
        plan, *_, reason = cake_backend.plan_atomic_gemm(
            *operands(T, d_in, d_out), **kw
        )
        assert plan is not None and reason == "", (d_in, d_out, T, reason)
        assert (plan.template, plan.splits, plan.chains, plan.split_order) == (
            template,
            splits,
            chains,
            order,
        )
        assert plan.reproducible
        assert plan.workspace == (splits > 1)
        # the plan is deterministic per shape, also where its bits are not
        again = cake_backend.plan_atomic_gemm(*operands(T, d_in, d_out), **kw)[0]
        assert (
            again.template,
            again.block_n,
            again.splits,
            again.chains,
            again.split_order,
        ) == (plan.template, plan.block_n, plan.splits, plan.chains, plan.split_order)
    # 148-SM sm_100a: BF16 4096 -> 1024 runs one split of two chains at either K-block parity, from 961 tokens
    _register_k1(monkeypatch, "sm_100a")
    kw100 = dict(transposed_out=True, arch="sm_100a", sm_count=148)
    for T in (961, 1024, 4097, 8192, 16170, 16171, 16172, 16231, 16300, 16384):
        X, G, out = operands(T, 4096, 1024)
        plan, *_, reason = cake_backend.plan_atomic_gemm(
            X, G, out.to(torch.bfloat16), **kw100
        )
        assert plan is not None and reason == "", (T, reason)
        assert (plan.template, plan.splits, plan.chains, plan.split_order) == (
            "dense_proj_gemm_atomic_nn_n256_t_c2",
            1,
            2,
            "arrival",
        )
        assert plan.reproducible and not plan.workspace
    # below the range (960 tokens, 15 K blocks): no two-chain plan
    X, G, out = operands(960, 4096, 1024)
    plan = cake_backend.plan_atomic_gemm(X, G, out.to(torch.bfloat16), **kw100)[0]
    assert plan is None or plan.chains == 1
    # outside the ranges: the split tables' choice
    for d_in, d_out, T in (
        (4096, 32, 16171),
        (4096, 32, 16385),
        (4096, 1024, 16171),
        (5120, 1024, 16385),
        (5120, 512, 16300),
    ):
        plan = cake_backend.plan_atomic_gemm(*operands(T, d_in, d_out), **kw)[0]
        assert plan is None or plan.split_order == "arrival", (d_in, d_out, T)
        assert plan is None or plan.chains in (1, 3), (d_in, d_out, T)


#: The ranges of :data:`K1_PARITY_RANGES_BY_SM`: ``(arch, sm_count) -> {key: ((lo, hi, even, odd), ...)}``; a program
#: ``(block_n, splits, chains, order)``, ``()`` the deterministic program, ``None`` the other tables.
_P2_128, _P4_128 = (128, 1, 2, "arrival"), (128, 1, 4, "arrival")
_P2_192, _P4_192 = (192, 1, 2, "arrival"), (192, 1, 4, "arrival")
_P2_256, _P8_256 = (256, 1, 2, "arrival"), (256, 1, 8, "arrival")
_P4_256 = (256, 1, 4, "arrival")
# fmt: off
_PARITY_BASE = {
    ("sm_107a", 1, 4096, 512, "f32"): (
        (961, 12000, _P2_128, _P4_128), (12001, 32767, (), _P4_128), (32768, 40961, (), ()),
    ),
    ("sm_107a", 1, 5120, 512, "f32"): ((961, 40961, _P2_128, _P4_128),),
    ("sm_107a", 1, 4096, 1024, "f32"): ((961, 40961, _P2_192, _P4_192),),
    ("sm_107a", 1, 5120, 1024, "f32"): ((961, 40961, _P2_256, _P8_256),),
}
_PARITY_212 = {
    ("sm_107a", 1, 4096, 512, "f32"): (
        (961, 12000, _P2_128, _P4_128), (12001, 24576, (), _P4_128), (24577, 40961, (), ()),
    ),
    ("sm_107a", 1, 5120, 512, "f32"): ((961, 40961, _P2_128, _P4_128),),
    ("sm_107a", 1, 4096, 1024, "f32"): (
        (961, 24575, _P2_192, _P4_192), (24576, 40961, (), _P4_192), (40962, 65536, None, _P4_192),
    ),
    ("sm_107a", 1, 5120, 1024, "f32"): (
        (961, 16169, _P2_256, _P8_256), (16170, 16172, _P2_256, _P4_256), (16173, 40961, _P2_256, _P8_256),
    ),
}
_PARITY_216 = {
    **_PARITY_BASE,
    ("sm_107a", 1, 4096, 1024, "f32"): ((961, 40961, _P2_192, _P4_192), (40962, 65536, None, _P4_192)),
}
_PARITY_208 = {
    **_PARITY_BASE,
    ("sm_107a", 1, 5120, 512, "f32"): ((961, 40961, _P2_128, _P4_128), (40962, 65536, (), ())),
    ("sm_107a", 1, 4096, 512, "f32"): (
        (961, 12000, _P2_128, _P4_128), (12001, 16171, (), _P4_128), (16172, 16230, _P2_128, _P4_128),
        (16231, 16231, (), ()), (16232, 16383, _P2_128, _P4_128), (16384, 16384, (), ()),
        (16385, 32767, (), _P4_128), (32768, 40961, (), ()),
    ),
    ("sm_107a", 1, 4096, 1024, "f32"): (
        (961, 16230, _P2_192, _P4_192), (16231, 16231, (), ()), (16232, 16383, _P2_192, _P4_192),
        (16384, 16384, (), ()), (16385, 40961, _P2_192, _P4_192), (40962, 65536, None, _P4_192),
    ),
}
K1_PARITY_ROWS = {
    ("sm_107a", 208): _PARITY_208,
    ("sm_107a", 212): _PARITY_212,
    ("sm_107a", 216): _PARITY_216,
}
# fmt: on


def _wgrad_operands(T, d_in, d_out):
    """``(X.T, G, out)`` of a weight gradient on CPU tensors (never written: nothing is launched)."""
    G = torch.empty(T, d_out, dtype=torch.bfloat16)
    X = torch.empty(T, d_in, dtype=torch.bfloat16)
    return X.t(), G, torch.empty(d_out, d_in, dtype=torch.float32)


def test_k1_parity_ranges_plan_their_program(monkeypatch):
    """:data:`K1_PARITY_RANGES_BY_SM` holds the measured ranges; at both ends of every range, one token inside them and
    its middle, :func:`k1_parity_plan` names the program of the token count's K-block parity (two chains at an even
    count, four at an odd one -- eight at ``5120 -> 1024`` --, the deterministic program or the other tables), and
    :func:`plan_atomic_gemm` plans that program (template, tile width, one split, chain count, a plan that repeats, no
    workspace) or keeps the row on the deterministic program; one token below the first range, an SM count without
    ranges, another output dtype and a batch keep the other tables; a part of more SMs than 216 takes the 216-SM ranges
    (:data:`K1_PARITY_RANGES_ABOVE_SM`)."""
    assert cake_backend.K1_PARITY_RANGES_BY_SM == K1_PARITY_ROWS
    assert cake_backend.K1_PARITY_RANGES_ABOVE_SM == {"sm_107a": 216}
    arch = "sm_107a"
    _register_k1(monkeypatch, arch)
    _register(
        monkeypatch,
        arch,
        *(
            cake_backend.atomic_k1_template(bn, True, c)
            for bn in (128, 192, 256)
            for c in (2, 4)
        ),
        cake_backend.atomic_k1_template(256, True, 8),
    )
    for (_, sms), rows in K1_PARITY_ROWS.items():
        kw = dict(transposed_out=True, arch=arch, sm_count=sms)
        for (_, _, d_in, d_out, _), ranges in rows.items():
            pk = dict(arch=arch, L=1, M=d_in, N=d_out, out_f32=True, sm_count=sms)
            for lo, hi, even, odd in ranges:
                points = {lo, lo + 1, (lo + hi) // 2, hi - 1, hi}
                for T in sorted(t for t in points if lo <= t <= hi):
                    want = odd if -(-T // 64) % 2 else even
                    what = (sms, d_in, d_out, T)
                    assert cake_backend.k1_parity_plan(K=T, **pk) == want, what
                    if want is None:
                        continue
                    plan, *_, reason = cake_backend.plan_atomic_gemm(
                        *_wgrad_operands(T, d_in, d_out), **kw
                    )
                    if want == ():
                        assert plan is None and "deterministic" in reason, what
                        continue
                    bn, splits, chains, order = want
                    assert plan is not None and reason == "", (what, reason)
                    assert (
                        plan.template,
                        plan.block_n,
                        plan.splits,
                        plan.chains,
                        plan.split_order,
                    ) == (
                        cake_backend.atomic_k1_template(bn, True, chains, order),
                        bn,
                        splits,
                        chains,
                        order,
                    ), what
                    assert plan.reproducible and not plan.workspace, what
            assert cake_backend.k1_parity_plan(K=ranges[0][0] - 1, **pk) is None
    w512 = dict(arch=arch, L=1, M=4096, N=512, K=16172, out_f32=True)
    for sms in (148, 152, 204, 210, 214):
        assert cake_backend.k1_parity_plan(sm_count=sms, **w512) is None, sms
    for sms in (217, 224, 264):
        for (_, _, d_in, d_out, _), ranges in K1_PARITY_ROWS[("sm_107a", 216)].items():
            for lo, hi, _even, _odd in ranges:
                for T in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1):
                    pk = dict(arch=arch, L=1, M=d_in, N=d_out, K=T, out_f32=True)
                    assert cake_backend.k1_parity_plan(
                        sm_count=sms, **pk
                    ) == cake_backend.k1_parity_plan(sm_count=216, **pk), (sms, T)
    assert (
        cake_backend.k1_parity_plan(sm_count=212, **dict(w512, out_f32=False)) is None
    )
    assert cake_backend.k1_parity_plan(sm_count=212, **dict(w512, L=2)) is None


#: Token counts at which :func:`test_k1_sm107a_fp32_wgrad_rows_run_no_one_or_three_chains` plans the four FP32 rows:
#: the ends of the ranges of every table that routes them on 208, 212 and 216 SMs, one token on each side, and the
#: token counts of the accuracy sweep.
# fmt: off
_PARITY_SWEEP_T = sorted({
    t + d
    for t in (961, 1024, 2048, 4096, 7680, 8192, 12000, 12001, 16172, 16192, 16231, 16256, 16300, 16320, 16384,
              16170, 20000, 24577, 32767, 32768, 36000, 40961, 40962, 45057, 49153, 65536)
    for d in (-1, 0, 1)
    if t + d >= 961
})
# fmt: on


@pytest.mark.parametrize("sms", [208, 212, 216, 224])
def test_k1_sm107a_fp32_wgrad_rows_run_no_one_or_three_chains(sms):
    """On ``sm_107a`` parts of 208, 212, 216 and more SMs (224: the 216-SM ranges, :data:`K1_PARITY_RANGES_ABOVE_SM`)
    the FP32 weight gradients ``4096 / 5120 -> 512 / 1024`` run, from 961 tokens up, no one-split program of one or
    three accumulation chains (one chain missed the accuracy check on all six input kinds, three chains on mirrored
    cancelling inputs): every token count of :data:`_PARITY_SWEEP_T` runs the program :func:`k1_parity_plan` names for
    it where it names one (one split of two, four or eight chains at the range's tile width), the deterministic program
    where it names that, and else the deterministic program or two splits; every program planned is one this checkout
    registers for ``sm_107a``."""
    kw = dict(transposed_out=True, arch="sm_107a", sm_count=sms)
    routed = 0
    for d_in, d_out in ((4096, 512), (5120, 512), (4096, 1024), (5120, 1024)):
        for T in _PARITY_SWEEP_T:
            plan, *_, reason = cake_backend.plan_atomic_gemm(
                *_wgrad_operands(T, d_in, d_out), **kw
            )
            want = cake_backend.k1_parity_plan(
                arch="sm_107a", L=1, M=d_in, N=d_out, K=T, out_f32=True, sm_count=sms
            )
            what = (sms, d_in, d_out, T, want, reason)
            if plan is None:
                assert want in (None, ()) and "deterministic" in reason, what
                continue
            assert want != (), what
            routed += 1
            assert plan.template in cake_jit.KERNELS["sm_107a"], what
            if want is not None:
                assert (
                    plan.block_n,
                    plan.splits,
                    plan.chains,
                    plan.split_order,
                ) == want, what
            if plan.splits > 1:
                assert (plan.splits, plan.chains) == (2, 1), what
                continue
            assert plan.chains in (2, 4, 8), what
            assert plan.reproducible, what
    assert routed


@pytest.mark.parametrize("sm_count", [208, 212, 216, 224])
def test_k1_parity_ranges_bind_per_token_count(monkeypatch, sm_count):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the ranges of
    :data:`K1_PARITY_RANGES_BY_SM` hold there too: on ``sm_107a`` parts of 208, 212, 216 and 224 SMs the weight gradient
    of each FP32 row binds, at both ends of each of its ranges and at an even and an odd token count inside it, the
    program of the token count's parity (the 216-SM ranges on 224 SMs), or the deterministic program, each token count
    its own cache entry."""
    arch = "sm_107a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sm_count, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    rows = K1_PARITY_ROWS[("sm_107a", min(sm_count, 216))]
    bf16 = torch.bfloat16
    bound = 0
    for (_, _, d_in, d_out, _), ranges in rows.items():
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in, dtype=torch.float32)
        cases = set()
        for lo, hi, even, odd in ranges:
            mid = (lo + hi) // 2
            for T in {lo, hi, mid, mid + 64}:
                if lo <= T <= hi:
                    cases.add((T, odd if -(-T // 64) % 2 else even))
        for T, want in sorted(cases):
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (sm_count, d_in, d_out, T, want)
            assert entry.k1 is not None, where
            if want == ():
                assert entry.plan is None and entry.det_plan is not None, where
                continue
            if want is None:
                continue
            assert entry.det_plan is None, where
            plan = entry.plan
            assert (plan.block_n, plan.splits, plan.chains, plan.split_order) == want, (
                where
            )
            bound += 1
        assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
        assert len(entry_fn.calls) == len(cases)
    assert bound


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_k1_program_rows_bind_their_program(monkeypatch, arch):
    """The forward and input-gradient entry points run, for a row of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` /
    :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` on its part, the table's program of this package with the program's argument
    order (no cached deterministic plan); one token below the row (where that is not a row of the table too, as inside
    a token window), an FP32 forward output and an operand without a unit inner stride keep the deterministic routing
    (``_k1_*_program_of`` is ``None``).  In a token window the rows at :data:`_KD_WINDOW_SAMPLE` run."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16, f32 = torch.bfloat16, torch.float32
    for (K, N, T), template in sorted(K1_FORWARD_PROGRAM_ROWS[(arch, 152)].items()):
        X, W, out = (
            torch.empty(T, K, dtype=bf16),
            torch.empty(N, K, dtype=bf16),
            torch.empty(T, N, dtype=bf16),
        )
        module = cake_jit.KERNELS[arch][template]
        assert cake_backend._k1_forward_program_of(X, W, out) == module
        n = len(entry_fn.calls)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, out)
            is out
        )
        assert len(entry_fn.calls) == n + 1 and not cake_backend._K1_DET_CACHE
        record = cake_jit.MODULES[module]
        items = -(-N // 512) * -(-T // record["build"]["kwargs"]["tile_n"])
        grid = (2 * min(76, items), 1, 1)
        values = dict(
            A=W.unsqueeze(0),
            B=X.unsqueeze(0),
            C=out.unsqueeze(0),
            M=N,
            N=T,
            K=K,
            L=1,
            enable_pdl=1,  # the default on a part the package serves
        )
        plan = cake_jit.arg_plan_of(record)
        want = cake_backend._order_atomic_arguments(module, plan, values, grid)
        _same_k1_arguments(list(entry_fn.calls[-1][0]), want, plan)
        assert cake_backend._k1_forward_program_of(X[:-1], W, out[:-1]) is None
        assert (
            cake_backend._k1_forward_program_of(X, W, torch.empty(T, N, dtype=f32))
            is None
        )
        assert (
            cake_backend._k1_forward_program_of(
                torch.empty(K, T, dtype=bf16).t(), W, out
            )
            is None
        )
    for (K, N, T, _), template in sorted(K1_DGRAD_PROGRAM_ROWS[(arch, 152)].items()):
        if T in _KD_WINDOW and T not in _KD_WINDOW_SAMPLE:
            continue
        G, W, out = (
            torch.empty(T, N, dtype=bf16),
            torch.empty(N, K, dtype=bf16),
            torch.empty(T, K, dtype=f32),
        )
        module = cake_jit.KERNELS[arch][template]
        assert cake_backend._k1_dgrad_program_of(G, W, out) == module
        n = len(entry_fn.calls)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out) is out
        )
        assert len(entry_fn.calls) == n + 1 and not cake_backend._K1_DET_CACHE
        record = cake_jit.MODULES[module]
        grid = (2 * min(76, -(-T // 256) * -(-K // 256)), 1, 1)
        spare = det_backend._dummy_map(out.device, bf16)
        values = dict(A=G, B=W.unsqueeze(0), out=det_backend._dummy(out.device, bf16), out32=out, C16=spare,
                      C32=out.unsqueeze(0), M=T, N=K, K=N, ldo=K, enable_pdl=1)  # fmt: skip
        plan = cake_jit.arg_plan_of(record)
        want = cake_backend._order_atomic_arguments(module, plan, values, grid)
        _same_k1_arguments(list(entry_fn.calls[-1][0]), want, plan)
        if (K, N, T - 1, "f32") not in K1_DGRAD_PROGRAM_ROWS[(arch, 152)]:
            assert cake_backend._k1_dgrad_program_of(G[:-1], W, out[:-1]) is None
        assert (
            cake_backend._k1_dgrad_program_of(G, W, torch.empty(T, K, dtype=bf16))
            is None
        )
    # another SM count: the deterministic routing
    facts = dataclasses.replace(det_backend.device_facts(), sm_count=151)
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
    K, N, T = 4096, 512, 16231
    X, W, out = (
        torch.empty(T, K, dtype=bf16),
        torch.empty(N, K, dtype=bf16),
        torch.empty(T, N, dtype=bf16),
    )
    assert cake_backend._k1_forward_program_of(X, W, out) is None


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_k1_program_rows_cache_their_binding(monkeypatch, arch):
    """A forward / input-gradient row of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` / :data:`K1_DGRAD_PROGRAM_RULES_BY_SM`
    binds its program once per key in :data:`_K1_PROGRAM_CACHE` (the key of :data:`_K1_DET_CACHE`: the entry point's
    flags, the PDL flag, the device, the stream and the geometry of its tensors); a repeat call -- on the same
    tensors or on other tensors of that geometry -- launches the bound program without the table lookup, with the
    call's own views in the program's order; a call of another PDL flag, device index, stream or output stride binds
    a new entry, an FP32 forward output is no program row and no key of a BF16 one, and ``_drop_convenience_entries``
    drops the entries."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    index = {"value": 0}
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: index["value"])
    bf16, f32 = torch.bfloat16, torch.float32
    cache = cake_backend._K1_PROGRAM_CACHE
    (K, N, T), template = min(K1_FORWARD_PROGRAM_ROWS[(arch, 152)].items())
    module = cake_jit.KERNELS[arch][template]
    plan = cake_jit.arg_plan_of(cake_jit.MODULES[module])
    lookup = cake_backend._k1_forward_program_of

    def forward(X, W, out, **kw):
        n = len(entry_fn.calls)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(
                X, W, out, **kw
            )
            is out
        )
        assert len(entry_fn.calls) == n + 1
        return list(entry_fn.calls[-1][0])

    def fresh():
        return (
            torch.empty(T, K, dtype=bf16),
            torch.empty(N, K, dtype=bf16),
            torch.empty(T, N, dtype=bf16),
        )

    X, W, out = fresh()
    first = forward(X, W, out)
    ((key, entry),) = cache.items()
    assert key[2] == (("fwd",), 1, False) and not cake_backend._K1_DET_CACHE
    assert (
        entry.module_name == module and entry.k1 is not None and entry.entry is not None
    )

    def no_lookup(*args, **kwargs):
        raise AssertionError("a cached program row looked its table up")

    monkeypatch.setattr(cake_backend, "_k1_forward_program_of", no_lookup)
    _same_k1_arguments(forward(X, W, out), first, plan)

    def layout(args):
        return [
            (tuple(a.shape), a.stride(), a.dtype) if isinstance(a, torch.Tensor) else a
            for a in args
        ]

    X2, W2, out2 = fresh()
    got = forward(X2, W2, out2)
    assert layout(got) == layout(first)
    pointers = {t.data_ptr() for t in got if isinstance(t, torch.Tensor)}
    assert {X2.data_ptr(), W2.data_ptr(), out2.data_ptr()} <= pointers
    assert not pointers & {X.data_ptr(), W.data_ptr(), out.data_ptr()}
    assert len(cache) == 1
    monkeypatch.setattr(cake_backend, "_k1_forward_program_of", lookup)
    # another PDL flag, device index, stream, output stride: a new entry each
    pdl = forward(X, W, out, enable_pdl=False)
    assert pdl[[name for _, name in plan].index("enable_pdl")] == 0
    assert len(cache) == 2
    index["value"] = 1
    forward(X, W, out)
    assert len(cache) == 3
    index["value"] = 0
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 7)
    forward(X, W, out)
    assert len(cache) == 4 and {k[1] for k in cache} == {0, 7}
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 0)
    padded = torch.empty(T, N + 64, dtype=bf16)[:, :N]
    assert lookup(X, W, padded) == module
    got = forward(X, W, padded)
    assert len(cache) == 5
    c = [name for _, name in plan].index("C")
    assert (got[c].data_ptr(), tuple(got[c].stride())) == (
        padded.data_ptr(),
        (T * (N + 64), N + 64, 1),
    )
    # an FP32 output is no program row (the deterministic routing) and no key of a BF16 one
    out32 = torch.empty(T, N, dtype=f32)
    assert lookup(X, W, out32) is None
    assert cake_backend._k1_key(0, 0, key[2], X, W, out32) != key
    rows = sorted(K1_DGRAD_PROGRAM_ROWS[(arch, 152)].items())
    if rows:
        (K, N, T, _), template = rows[0]
        G, W, out = (
            torch.empty(T, N, dtype=bf16),
            torch.empty(N, K, dtype=bf16),
            torch.empty(T, K, dtype=f32),
        )
        module = cake_jit.KERNELS[arch][template]
        plan = cake_jit.arg_plan_of(cake_jit.MODULES[module])
        n = len(entry_fn.calls)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out) is out
        )
        first = list(entry_fn.calls[-1][0])
        assert len(cache) == 6 and next(reversed(cache))[2] == (("dgrad",), 1, False)
        assert next(reversed(cache.values())).module_name == module
        monkeypatch.setattr(cake_backend, "_k1_dgrad_program_of", no_lookup)
        G2 = torch.empty(T, N, dtype=bf16)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G2, W, out)
            is out
        )
        assert len(entry_fn.calls) == n + 2 and len(cache) == 6
        got = list(entry_fn.calls[-1][0])
        assert layout(got) == layout(first)
        a = [name for _, name in plan].index("A")
        assert got[a].data_ptr() == G2.data_ptr()
    cake_backend._drop_convenience_entries()
    assert not cache


_KD_WINDOW_PROGRAMS = {
    ("sm_100a", 148): {
        (4096, 512): _KD_TMA_LEAN_FREE,
        (5120, 512): _KD_TMA_LEAN_FREE,
        (6144, 576): _KD_TMA_LEAN_FREE,
    },
    ("sm_103a", 152): {
        (4096, 512): _KD_TMA_LEAN_FREE,
        (5120, 512): _KD_TMA_LEAN_FREE,
        (6144, 576): _KD_TMA_LEAN_FREE,
    },
}


@pytest.mark.parametrize("arch, sms", sorted(_KD_WINDOW_PROGRAMS))
def test_k1_dgrad_window_rows_bind_their_program(monkeypatch, arch, sms):
    """On a 148-SM ``sm_100a`` and a 152-SM ``sm_103a`` part the input-gradient entry point runs, at the first, an
    inner and the last token count of the window of :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` (16172 to 16384 tokens), the
    window's program -- ``dense_proj_gemm_dgrad_kn_n256_f32_tma_lean_free`` for ``4096 -> 512``, ``5120 -> 512`` and
    ``6144 -> 576`` -- with the program's argument order (no cached deterministic plan); one token outside the window
    keeps the deterministic routing."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    facts = dataclasses.replace(det_backend.device_facts(), sm_count=sms)
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
    bf16, f32 = torch.bfloat16, torch.float32
    rows = K1_DGRAD_PROGRAM_ROWS[(arch, sms)]

    def views(T, N, K):
        return (
            torch.empty(T, N, dtype=bf16),
            torch.empty(N, K, dtype=bf16),
            torch.empty(T, K, dtype=f32),
        )

    for (K, N), template in _KD_WINDOW_PROGRAMS[(arch, sms)].items():
        window = {key[2] for key in rows if key[:2] == (K, N)}
        assert window == set(_KD_WINDOW), (K, N)
        assert {rows[(K, N, T, "f32")] for T in window} == {template}
        module = cake_jit.KERNELS[arch][template]
        for T in (16172, 16300, 16384):
            G, W, out = views(T, N, K)
            assert cake_backend._k1_dgrad_program_of(G, W, out) == module
            n = len(entry_fn.calls)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out)
                is out
            )
            assert len(entry_fn.calls) == n + 1 and not cake_backend._K1_DET_CACHE
            grid = (2 * min(sms // 2, -(-T // 256) * -(-K // 256)), 1, 1)
            spare = det_backend._dummy_map(out.device, bf16)
            values = dict(A=G, B=W.unsqueeze(0), out=det_backend._dummy(out.device, bf16), out32=out, C16=spare,
                          C32=out.unsqueeze(0), M=T, N=K, K=N, ldo=K, enable_pdl=1)  # fmt: skip
            plan = cake_jit.arg_plan_of(cake_jit.MODULES[module])
            want = cake_backend._order_atomic_arguments(module, plan, values, grid)
            _same_k1_arguments(list(entry_fn.calls[-1][0]), want, plan)
        for T in (16171, 16385):
            G, W, out = views(T, N, K)
            assert cake_backend._k1_dgrad_program_of(G, W, out) is None, (K, N, T)


@pytest.mark.parametrize("arch, sms", sorted(K1_WGRAD_SK_ROWS))
def test_k1_wgrad_sk_rows_bind_their_program(monkeypatch, arch, sms):
    """The weight-gradient entry point runs, for a row of :data:`K1_WGRAD_SK_RULES_BY_SM` on its part (148-SM
    sm_100a, 152-SM sm_103a, 212-SM sm_107a), the stream-K program with one unit per CTA pair (74, 76, 106), the grid
    one CTA pair per unit, the unit length the line's K
    blocks over the units, the arena's slots and flag words, and the program's argument order; a non-default ``sk`` or
    ``cta_rows``, another SM count and one token off the row keep the other tables."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    facts = dataclasses.replace(det_backend.device_facts(), sm_count=sms)
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
    bf16 = torch.bfloat16
    units = sms // 2
    for (d_in, d_out, T, dt), template in sorted(K1_WGRAD_SK_ROWS[(arch, sms)].items()):
        G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
        out = torch.empty(d_out, d_in, dtype=torch.float32 if dt == "f32" else bf16)
        module = cake_jit.KERNELS[arch][template]
        assert cake_backend._k1_wgrad_sk_program_of(G, X, out, "auto", None) == module
        assert cake_backend._k1_wgrad_sk_program_of(G, X, out, True, None) is None
        assert cake_backend._k1_wgrad_sk_program_of(G, X, out, "auto", 128) is None
        assert (
            cake_backend._k1_wgrad_sk_program_of(G[:-1], X[:-1], out, "auto", None)
            is None
        )
        n = len(entry_fn.calls)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out) is out
        )
        assert len(entry_fn.calls) == n + 1 and not cake_backend._K1_ATOMIC_CACHE
        line = (d_in // 256) * (d_out // 256) * -(-T // 64)
        arena = cake_backend._ATOMIC_ARENAS[(0, 0)]
        values = dict(A=X.unsqueeze(0), B=G.unsqueeze(0), out=out, ws=arena.buffers["k1_slots"],
                      flags=arena.buffers["k1_tickets"], M=d_in, N=d_out, K=T, num_l=1, num_units=units,
                      unit_len=-(-line // units), ldo=d_in, out_l=d_out * d_in, enable_pdl=0)  # fmt: skip
        plan = cake_jit.arg_plan_of(cake_jit.MODULES[module])
        want = cake_backend._order_atomic_arguments(module, plan, values, (sms, 1, 1))
        _same_k1_arguments(list(entry_fn.calls[-1][0]), want, plan)
    assert int(arena.buffers["k1_tickets"].view(torch.int32).count_nonzero()) == 0
    facts = dataclasses.replace(facts, sm_count=sms + 4)
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
    assert cake_backend._k1_wgrad_sk_program_of(G, X, out, "auto", None) is None


def test_k1_entry_points_go_from_their_key_to_the_bound_launch(monkeypatch):
    """Once a key is bound, the K1 entry points launch it straight from the key
    (``_k1_cached_launch``): neither ``_det_k1`` nor ``_atomic_k1`` runs, and the launch gets the arguments their
    launch of the key gets. An ``out`` of the same geometry is the same key and launches into itself; another geometry
    is another key and takes the planning path. Each thread makes the stream context of a (device, stream) once and
    reuses it."""
    entry_fn = _cpu_k1_launches(monkeypatch, "sm_103a")
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    made = []

    def stream_context(index, raw=None):
        made.append((threading.get_ident(), index, raw))
        return contextlib.nullcontext()

    monkeypatch.setattr(det_backend, "_ffi_stream_context", stream_context)
    bf16, T = torch.bfloat16, 1000
    X, W, G = (
        torch.empty(T, 4096, dtype=bf16),
        torch.empty(128, 4096, dtype=bf16),
        torch.empty(T, 128, dtype=bf16),
    )
    out_f, out_d, out_w = (
        torch.empty(T, 128, dtype=bf16),
        torch.empty(T, 4096, dtype=bf16),
        torch.empty(128, 4096),
    )

    def calls():
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, out_f)
            is out_f
        )
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out_d)
            is out_d
        )
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w)
            is out_w
        )

    def spelled(args):
        return [
            (a.data_ptr(), tuple(a.shape), a.stride(), a.dtype)
            if isinstance(a, torch.Tensor)
            else a
            for a in args
        ]

    # the first round plans and grows the arena; the second takes the planning path's cached launches
    calls()
    cached_launch = cake_backend._k1_cached_launch
    monkeypatch.setattr(cake_backend, "_k1_cached_launch", lambda *args: False)
    entry_fn.calls.clear()
    calls()
    planned = [spelled(args) for args, _ in entry_fn.calls]
    assert len(planned) == 3 and len(made) == 1

    def planning_path(*args, **kwargs):
        raise AssertionError("planning path")

    monkeypatch.setattr(cake_backend, "_k1_cached_launch", cached_launch)
    monkeypatch.setattr(cake_backend, "_det_k1", planning_path)
    monkeypatch.setattr(cake_backend, "_atomic_k1", planning_path)
    entry_fn.calls.clear()
    calls()
    assert [spelled(args) for args, _ in entry_fn.calls] == planned
    assert len(made) == 1
    other = torch.empty_like(out_f)
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, other)
        is other
    )
    (args, _) = entry_fn.calls[-1]
    pointers = {a.data_ptr() for a in args if isinstance(a, torch.Tensor)}
    assert other.data_ptr() in pointers and out_f.data_ptr() not in pointers
    with pytest.raises(AssertionError, match="planning path"):
        cake_backend.cake_dense_gemm_nondet_train_projection_forward(
            X[:999], W, out_f[:999]
        )
    # another thread makes its own context of the same (device, stream)
    entry_fn.calls.clear()
    worker = threading.Thread(target=calls)
    worker.start()
    worker.join()
    assert len(entry_fn.calls) == 3 and len(made) == 2 and made[0][0] != made[1][0]
    assert made[0][1:] == made[1][1:] == (0, 0)


def test_k1_sm107a_wide_fp32_rows_bind_per_token_count(monkeypatch, no_k1_chains):
    """The convenience path keys a K1 call by the geometry of its views, the token count included, and plans each key
    at the part's SM count: on a 216-SM ``sm_107a`` part the weight gradient of each row of
    ``K1_SM107A_WIDE_FP32_ROWS`` binds its one-split program one token below its ``K1_SM107A_WIDE_FP32_OPEN`` and
    the deterministic package's plan at that token count and at 65536 tokens, after the smaller token count's entry
    is cached."""
    arch = "sm_107a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=216, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    for (d_in, d_out), choice in K1_SM107A_WIDE_FP32_ROWS.items():
        T0 = K1_SM107A_WIDE_FP32_OPEN[(d_in, d_out)]
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in)
        for T in (T0 - 1, T0, 65536):
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (d_in, d_out, T)
            assert entry.k1 is not None, where
            if T < T0:
                assert entry.det_plan is None, where
                assert (entry.plan.block_n, entry.plan.splits) == choice, where
            else:
                assert entry.plan is None and entry.det_plan is not None, where
        assert len(cake_backend._K1_ATOMIC_CACHE) == 3
        assert len(entry_fn.calls) == 3


def test_k1_212_sm_token_ranges_bind_per_token_count(monkeypatch):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the ranges of
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` hold there too: on a 212-SM ``sm_107a`` part the weight gradient of
    each row of ``K1_212_TOKEN_RANGES`` binds the row's 212-SM program at the first and the last token count of its
    ranges and the deterministic package's plan one token outside them (outside the window of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS`), each token count its own cache entry."""
    arch = "sm_107a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=212, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    table = cake_backend.K1_SPLIT_RULES_BY_SM[(arch, 212)]
    bf16 = torch.bfloat16
    for (d_in, d_out, dt), spans in K1_212_TOKEN_RANGES.items():
        choice = table[(arch, 1, d_in, d_out, dt)]
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in, dtype=torch.float32 if dt == "f32" else bf16)
        cases = []
        for lo, hi in spans:
            cases += [(lo, True), (hi, True)]
            cases += [
                (T, False)
                for T in (lo - 1, hi + 1)
                if not first <= T <= last and not any(a <= T <= b for a, b in spans)
            ]
        cases = sorted(set(cases))
        for T, routed in cases:
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (d_in, d_out, dt, T)
            assert entry.k1 is not None, where
            if routed:
                assert entry.det_plan is None, where
                assert (entry.plan.block_n, entry.plan.splits) == choice, where
            else:
                assert entry.plan is None and entry.det_plan is not None, where
        assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
        assert len(entry_fn.calls) == len(cases)


def test_k1_212_sm_range_overrides_bind_per_token_count(monkeypatch):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the ranges of
    :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` hold there too: on a 212-SM ``sm_107a`` part the weight gradient of each
    row of ``K1_212_RANGE_OVERRIDES`` binds that range's program at its first and last token count and the
    deterministic package's plan one token outside it, each token count its own cache entry."""
    arch = "sm_107a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=212, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    for (d_in, d_out, dt), spans in K1_212_RANGE_OVERRIDES.items():
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in, dtype=torch.float32 if dt == "f32" else bf16)
        cases = []
        for lo, hi, choice in spans:
            cases += [(lo, tuple(choice)), (hi, tuple(choice))]
            cases += [
                (T, None)
                for T in (lo - 1, hi + 1)
                if not any(a <= T <= b for a, b, _ in spans)
            ]
        cases = sorted(set(cases))
        for T, choice in cases:
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (d_in, d_out, dt, T)
            assert entry.k1 is not None, where
            if choice is not None:
                assert entry.det_plan is None, where
                assert (entry.plan.block_n, entry.plan.splits) == choice, where
            else:
                assert entry.plan is None and entry.det_plan is not None, where
        assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
        assert len(entry_fn.calls) == len(cases)


def test_k1_212_sm_two_split_ranges_bind_per_token_count(monkeypatch, no_k1_chains):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the ranges of
    :data:`K1_TWO_SPLIT_RANGES_BY_SM` hold there too: on a 212-SM ``sm_107a`` part the weight gradient of each row of
    ``K1_212_TWO_SPLIT_RANGES`` binds that range's two-split program at its first and last token count and one token
    past its first token count (an odd number of 64-token K blocks), and the deterministic package's plan one token
    outside it, each token count its own cache entry."""
    arch = "sm_107a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=212, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    for (d_in, d_out, dt), spans in K1_212_TWO_SPLIT_RANGES.items():
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in, dtype=torch.float32 if dt == "f32" else bf16)
        cases = []
        for lo, hi, choice in spans:
            assert _k1_blocks(lo + 1) % 2 == 1, (d_in, d_out, lo)
            two = tuple(choice)
            cases += [(lo, two), (hi, two), (lo + 1, two)]
            cases += [
                (T, None)
                for T in (lo - 1, hi + 1)
                if not any(a <= T <= b for a, b, _ in spans)
            ]
        cases = sorted(set(cases))
        for T, choice in cases:
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (d_in, d_out, dt, T)
            assert entry.k1 is not None, where
            if choice is not None:
                assert entry.det_plan is None, where
                assert (entry.plan.block_n, entry.plan.splits) == choice, where
            else:
                assert entry.plan is None and entry.det_plan is not None, where
        assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
        assert len(entry_fn.calls) == len(cases)


def test_k1_exact_sm_count_exclusions_bind_per_token_count(monkeypatch):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the entries of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` hold there too: on a 148-SM part the weight gradient of each row of
    ``K1_148_EXCLUDED_TOKENS`` binds the deterministic package's plan at the first and the last token count of a range
    (65537 for a range without an upper bound) and the row's :data:`K1_SPLIT_RULES` program one token outside it, each
    token count its own cache entry."""
    bf16 = torch.bfloat16
    for arch in sorted({k[0] for k in K1_148_EXCLUDED_TOKENS}):
        entry_fn = _cpu_k1_launches(monkeypatch, arch)
        facts = det_backend.DeviceFacts(
            index=0, arch=arch, sm_count=148, l2_bytes=126 << 20
        )
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
        for (a, d_in, d_out, dt), spans in K1_148_EXCLUDED_TOKENS.items():
            if a != arch:
                continue
            rule = tuple(cake_backend.K1_SPLIT_RULES[(arch, 1, d_in, d_out, dt)])
            cake_backend._K1_ATOMIC_CACHE.clear()
            entry_fn.calls.clear()
            out = torch.empty(d_out, d_in, dtype=torch.float32 if dt == "f32" else bf16)
            held = lambda T, spans=spans: any(  # noqa: E731
                lo <= T and (hi is None or hi >= T) for lo, hi in spans
            )
            cases = set()
            for lo, hi in spans:
                cases |= {(lo, None), (65537 if hi is None else hi, None)}
                cases |= {
                    (T, rule)
                    for T in (lo - 1, None if hi is None else hi + 1)
                    if T is not None and not held(T)
                }
            cases = sorted(cases)
            for T, choice in cases:
                G, X = (
                    torch.empty(T, d_out, dtype=bf16),
                    torch.empty(T, d_in, dtype=bf16),
                )
                assert (
                    cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
                        G, X, out
                    )
                    is out
                )
                entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
                where = (arch, d_in, d_out, dt, T)
                assert entry.k1 is not None, where
                if choice is not None:
                    assert entry.det_plan is None, where
                    assert (entry.plan.block_n, entry.plan.splits) == choice, where
                else:
                    assert entry.plan is None and entry.det_plan is not None, where
            assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
            assert len(entry_fn.calls) == len(cases)


@pytest.mark.parametrize(
    "arch, sm_count", [("sm_100a", 152), ("sm_103a", 150), ("sm_107a", 217)]
)
def test_k1_odd_block_exclusions_bind_per_token_count(monkeypatch, arch, sm_count):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the entries of
    :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` hold there too: for each FP32 row of the architecture in
    ``K1_ODD_BLOCK_EXCLUDED_TOKENS`` with an entry on the part, the weight gradient binds the row's two-split program
    one token below the first token count of its range (65537, or 1024 for sm_107a ``8192 -> 128`` on more than 216
    SMs) and one token past the end of that token count's block (an even block count), and the deterministic package's
    plan at that first token count and at the last token count of its block (an odd block count), each token count its
    own cache entry.  (On 152-SM sm_103a parts
    these rows run the deterministic program from 32768 tokens up: their entries of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`.)"""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sm_count, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    keys = sorted(
        k
        for k, entries in K1_ODD_BLOCK_EXCLUDED_TOKENS.items()
        if k[0] == arch and any(sm_count >= e[0] for e in entries)
    )
    assert keys
    for key in keys:
        _, _, d_in, d_out, dt = key
        assert dt == "f32"
        (lo,) = {e[1] for e in K1_ODD_BLOCK_EXCLUDED_TOKENS[key] if sm_count >= e[0]}
        rule = tuple(cake_backend.K1_SPLIT_RULES[key])
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in, dtype=torch.float32)
        last = 64 * _k1_blocks(lo)
        assert lo in (1025, 65538) and _k1_blocks(lo) % 2 == 1
        cases = ((lo - 1, rule), (lo, None), (last, None), (last + 1, rule))
        for T, choice in cases:
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (d_in, d_out, T)
            assert entry.k1 is not None, where
            if choice is not None:
                assert entry.det_plan is None, where
                assert (entry.plan.block_n, entry.plan.splits) == choice, where
            else:
                assert entry.plan is None and entry.det_plan is not None, where
        assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
        assert len(entry_fn.calls) == len(cases)


@pytest.mark.parametrize("sm_count", [212, 216])
def test_k1_chain_ranges_bind_per_token_count(monkeypatch, sm_count):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the ranges of
    :data:`K1_CHAIN_RANGES_BY_SM` hold there too: on a 212-SM and a 216-SM ``sm_107a`` part the weight gradient of each
    row of ``K1_CHAIN_RANGES`` binds the one-split program of three chains at the first and the last token count of a
    range and at 4097 tokens inside it (inside the row's 212-SM two-split range too), and the routing of the other
    tables at the measured token counts where the program failed (the deterministic program, or a program of one chain:
    the one-split program of ``5120 -> 1024`` at 32768 tokens on 216 SMs; the fixed-order program of
    :data:`K1_ORDERED_RANGES_BY_SM` where one of its ranges holds the token count: two chains of ``4096 -> 1024`` at
    16384 tokens and four at 16172 tokens on 212 SMs), each token count its own cache entry.
    :data:`K1_PARITY_RANGES_BY_SM`, which comes first from 961 tokens, is taken out
    (``test_k1_parity_ranges_bind_per_token_count`` binds it)."""
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    arch = "sm_107a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sm_count, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    for (d_in, d_out), spans in K1_CHAIN_RANGES[sm_count].items():
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in, dtype=torch.float32)
        cases = []
        for lo, hi, (block_n, chains) in spans:
            for T in (lo, hi, 4097):
                if lo <= T <= hi:
                    # a range of K1_ORDERED_RANGES_BY_SM comes first (16172 tokens: four chains on 212 SMs)
                    ordered = cake_backend.k1_ordered_plan(
                        arch=arch,
                        L=1,
                        M=d_in,
                        N=d_out,
                        K=T,
                        out_f32=True,
                        sm_count=sm_count,
                    )
                    cases.append(
                        (T, tuple(ordered[:3]) if ordered else (block_n, 1, chains))
                    )
        misses = K1_CHAIN_MISSES[sm_count].get((d_in, d_out), ())
        for T in misses:
            ordered = cake_backend.k1_ordered_plan(
                arch=arch, L=1, M=d_in, N=d_out, K=T, out_f32=True, sm_count=sm_count
            )
            if ordered is not None:
                cases.append((T, tuple(ordered[:3])))
                continue
            with monkeypatch.context() as m:
                m.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})
                other = cake_backend.k1_plan(
                    arch=arch,
                    L=1,
                    M=d_in,
                    N=d_out,
                    K=T,
                    out_f32=True,
                    sm_count=sm_count,
                )
            cases.append((T, None if other is None else (other[0], other[1], 1)))
        cases = sorted(set(cases), key=lambda c: c[0])
        assert any(c is not None and c[2] == 3 for _, c in cases)
        for T, choice in cases:
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            entry = next(reversed(cake_backend._K1_ATOMIC_CACHE.values()))
            where = (d_in, d_out, T)
            assert entry.k1 is not None, where
            if choice is not None:
                assert entry.det_plan is None, where
                plan = entry.plan
                assert (plan.block_n, plan.splits, plan.chains) == choice, where
                assert plan.template.endswith("_c3") == (choice[2] == 3), where
            else:
                assert entry.plan is None and entry.det_plan is not None, where
        assert len(cake_backend._K1_ATOMIC_CACHE) == len(cases)
        assert len(entry_fn.calls) == len(cases)


def test_k1_swapped_small_token_forward_stays_on_the_deterministic_call(monkeypatch):
    """A batched forward the deterministic package runs on its swapped small-token program (``_swap_ab_launch``) is
    not bound here: its entry stays unbound and ``cake_dense_gemm_nondet_train_projection`` makes the deterministic package's call, with
    ``enable_pdl`` and ``weights_static`` (part of the key); a batched forward of more tokens is bound."""
    arch = "sm_103a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    det = _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_projection")
    bf16, L, K, N = torch.bfloat16, 8, 4096, 1024

    def views(M):
        A = torch.empty(L, M, K, dtype=bf16)
        return (
            A,
            torch.empty(L, N, K, dtype=bf16).transpose(1, 2),
            torch.empty(L, M, N, dtype=bf16),
        )

    A, B, out = views(64)
    for weights_static in (False, True):
        launch = det_backend._swap_ab_launch(A, B, out, False, weights_static)
        assert launch is not None
        entry = cake_backend._plan_det_k1(
            A, B, out, False, (), (A, B, out), 0, 0, 1, weights_static
        )
        assert entry.plan is None and entry.k1 is None and entry.entry is None
        for _ in range(2):
            result = cake_backend.cake_dense_gemm_nondet_train_projection(
                A,
                B,
                out,
                enable_pdl=True,
                weights_static=weights_static,
            )
            assert result is det.result
    want = {"transposed_out": False, "enable_pdl": True}
    assert det.calls == 2 * [((A, B, out), want)] + 2 * [
        ((A, B, out), {**want, "weights_static": True})
    ]
    # ``weights_static`` is part of the key
    assert len(cake_backend._K1_DET_CACHE) == 2
    assert all(e.k1 is None for e in cake_backend._K1_DET_CACHE.values())
    entry_fn.calls.clear()
    A, B, out = views(1000)
    assert det_backend._swap_ab_launch(A, B, out, False) is None
    assert (
        cake_backend.cake_dense_gemm_nondet_train_projection(A, B, out, enable_pdl=True)
        is out
    )
    assert len(det.calls) == 4 and len(entry_fn.calls) == 1
    assert next(reversed(cake_backend._K1_DET_CACHE.values())).k1 is not None


def _router_spec_tokens(rule) -> set:
    tokens = {1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 224}
    tokens |= {4096, 8192, 32768, 32769}
    if isinstance(rule, tuple):
        for lo, hi, _ in rule:
            tokens |= {lo - 1, lo, lo + 1, hi - 1, hi, hi + 1}
    return {t for t in tokens if t >= 1}


def _router_launch_dims(layout, hidden, experts, T):
    """``(M, N, K)`` of the launched router GEMM of a row of the layout class ``layout`` at ``T`` tokens: the forward
    ``[T, experts] = [T, hidden] @ [hidden, experts]``, the input gradient ``[T, hidden] = [T, experts] @ [experts,
    hidden]``, the swapped weight gradient ``[hidden, experts] = [hidden, T] @ [T, experts]``."""
    return {
        "kk": (T, experts, hidden),
        "kn": (T, hidden, experts),
        "nn_t": (hidden, experts, T),
    }[layout]


def test_router_rule_spec_is_the_planners_verdict():
    """Where ``_router_rule_spec`` gives no spec on an SM count, ``router_partition`` without a split count runs the
    deterministic program there (the convenience path's pre-check then skips the planner exactly where the planner
    would keep the row); a spec it gives is the planner's (a head/tail or unit-count spec is the partition on an SM
    count whose router table holds, ``router_rules``)."""
    rows = dict(cake_backend.ROUTER_RULES)
    for arch in cake_backend.ROUTER_SM_COUNT:
        for layout in ("kk", "kn", "nn_t"):
            rows.setdefault((arch, layout, 2048, 256), None)  # rows without an entry
            rows.setdefault((arch, layout, 5120, 512), None)
    checked = 0
    for (arch, layout, hidden, experts), rule in rows.items():
        for T in _router_spec_tokens(rule):
            M, N, K = _router_launch_dims(layout, hidden, experts, T)
            for sm_count in (148, 152, 208, 212, 216):
                spec = cake_backend._router_rule_spec(
                    layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K
                )
                part = cake_backend.router_partition(
                    layout, arch=arch, M=M, N=N, K=K, sm_count=sm_count
                )
                if spec is None:
                    assert part is None, (arch, layout, hidden, experts, T, sm_count)
                elif spec != "waves" and part is not None:
                    m_tiles, n_tiles, k_iters = cake_backend._router_grid(M, N, K)
                    ksum = cake_backend.router_ksum_spec_splits(spec)
                    assert part == cake_backend.router_partition_of(
                        spec if ksum is None else f"S{ksum}",
                        m_tiles // 2,
                        n_tiles,
                        k_iters,
                        sm_count,
                    )
                checked += 1
    assert checked > 1000


def test_k2_rows_without_a_verdict_skip_the_planner(monkeypatch):
    """``_k2_row_without_verdict`` is true exactly for the valid ``kk`` / ``kn`` / ``nn_t`` bindings the planner (no
    split count) keeps on the deterministic program for want of a verdict, and false for every binding the planner
    refuses, plans or keeps for another reason (left to the planner); a verdict added at run time is seen."""
    arch, sm_count = "sm_103a", 152
    _register(
        monkeypatch,
        arch,
        "router_fp32_gemm_atomic_kk",
        "router_fp32_gemm_atomic_nn_t",
        "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX,
    )
    kept = planned = 0
    for hidden in (2048, 4096, 5120, 6144):
        for T in (64, 128, 192, 193, 224, 1024):
            X, W, G = (
                torch.empty(T, hidden),
                torch.empty(256, hidden),
                torch.empty(T, 256),
            )
            for A, B, out in (
                (X, W.t(), torch.empty(T, 256)),
                (G.t(), X, torch.empty(256, hidden)),
            ):
                keeps = cake_backend._k2_row_without_verdict(A, B, out, arch, sm_count)
                plan, reason = cake_backend.plan_atomic_router(
                    A, B, out, arch=arch, sm_count=sm_count
                )
                want = plan is None and reason == (
                    "the planner keeps the row on the deterministic program"
                )
                if keeps:
                    assert want, (hidden, T, A.shape)
                    kept += 1
                else:
                    layout = "kk" if A.stride(1) == 1 else "nn_t"
                    M, N = (T, 256) if layout == "kk" else (hidden, 256)
                    K = hidden if layout == "kk" else T
                    assert (
                        cake_backend._router_rule_spec(
                            layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K
                        )
                        is not None
                    ), (hidden, T, layout)
                    planned += plan is not None
    assert kept and planned
    # the kk class (the forward): past its SIMT windows (to 192 tokens on 152-SM sm_100a / sm_103a parts)
    X, W = torch.empty(224, 5120), torch.empty(256, 5120)
    out = torch.empty(224, 256)
    without_verdict = cake_backend._k2_row_without_verdict
    assert without_verdict(X, W.t(), out, arch, sm_count)
    assert not without_verdict(X[:64], W.t(), out[:64], arch, sm_count)
    assert without_verdict(X[:64], W.t(), out[:64], arch, 148)
    # bindings the planner refuses or classifies otherwise stay with it
    assert not without_verdict(X.bfloat16(), W.t(), out, arch, sm_count)
    assert not without_verdict(X, W.t(), out.double(), arch, sm_count)
    assert not without_verdict(X[None], W.t(), out, arch, sm_count)
    assert not without_verdict(X, W.t(), out[:, :128], arch, sm_count)
    assert not without_verdict(X[:0], W.t(), out[:0], arch, sm_count)
    assert not without_verdict(X, W.t()[:4096], out, arch, sm_count)
    # the kn class (the input gradient): a SIMT window up to 128 tokens on 152-SM sm_100a and sm_103a parts, none on
    # 148-SM parts or past the window
    g, w, dx = torch.empty(64, 256), torch.empty(256, 5120), torch.empty(64, 5120)
    assert not without_verdict(g, w, dx, arch, 152)
    assert not without_verdict(g, w, dx, "sm_100a", 152)
    assert without_verdict(g, w, dx, arch, 148)
    assert cake_backend.plan_atomic_router(g, w, dx, arch=arch, sm_count=148)[0] is None
    g2, dx2 = torch.empty(192, 256), torch.empty(192, 5120)
    assert without_verdict(g2, w, dx2, arch, 152)
    assert (
        cake_backend.plan_atomic_router(g2, w, dx2, arch=arch, sm_count=152)[0] is None
    )
    assert not without_verdict(
        torch.zeros(1).expand(224, 5120), W.t(), out, arch, sm_count
    )  # no unit stride
    monkeypatch.setitem(cake_backend.ROUTER_RULES, (arch, "kk", 5120, 256), "S2")
    assert not without_verdict(X, W.t(), out, arch, sm_count)


def test_router_tables_count_their_changes():
    """``ROUTER_RULES``, ``ROUTER_RULES_BY_SM`` and each table it holds count their changes: every way of changing
    a table moves the generation the convenience path's memo is kept for; a copy is a plain ``dict`` with the same
    items."""
    tables = (
        cake_backend.ROUTER_RULES,
        cake_backend.ROUTER_RULES_BY_SM,
        *cake_backend.ROUTER_RULES_BY_SM.values(),
    )
    assert all(type(t) is cake_backend._VerdictTable for t in tables)
    for t in tables:
        assert type(dict(t)) is dict and dict(t) == t
    key = ("sm_103a", "kk", 2048, 256)
    table = cake_backend._VerdictTable({("sm_103a", "nn_t", 2048, 256): None})
    changes = (
        lambda: table.__setitem__(key, "S2"),
        lambda: table.__delitem__(key),
        lambda: table.update({key: None}),
        lambda: table.pop(key),
        lambda: table.setdefault(key, "S2"),
        lambda: table.__ior__({key: None}),
        lambda: table.popitem(),
        lambda: table.clear(),
    )
    for change in changes:
        before = cake_backend._VERDICT_GENERATION
        change()
        assert before < cake_backend._VERDICT_GENERATION
    assert table == {}


def test_router_memo_follows_the_router_tables(monkeypatch):
    """The memo of the router tables (``_router_memo``) is kept while no table changes and derived again after any
    change: the forward verdicts of the 152-SM table and of the 212-SM sm_107a table ship, and with them cleared none is seen; a forward verdict set in any table, and its removal, are seen at the next call;
    a table rebound in place of one is seen, and one that does not count its changes (a plain ``dict``) is read on
    every call; ``cake_dense_gemm_nondet_train_convenience_cache_clear`` drops the memo."""
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    memo = cake_backend._router_memo()
    assert memo.tracked and memo.forward_verdicts and not memo.without_verdict
    assert cake_backend._router_memo() is memo
    # the shipped forward verdicts: the cluster SIMT windows of the 152-SM table and of the 212-SM sm_107a table;
    # without them no table gives one
    shipped = [
        k
        for k, rule in cake_backend.ROUTER_RULES.items()
        if k[1] == "kk" and rule is not None
    ]
    assert {k[0] for k in shipped} == {"sm_100a", "sm_103a"}
    shipped_by_sm = [
        (part, key)
        for part, table in cake_backend.ROUTER_RULES_BY_SM.items()
        for key, rule in table.items()
        if key[1] == "kk" and rule is not None
    ]
    assert {part for part, _ in shipped_by_sm} == {("sm_107a", 212)}
    for key in shipped:
        monkeypatch.setitem(cake_backend.ROUTER_RULES, key, None)
    for part, key in shipped_by_sm:
        monkeypatch.setitem(cake_backend.ROUTER_RULES_BY_SM[part], key, None)
    memo = cake_backend._router_memo()
    assert memo.tracked and not memo.forward_verdicts
    own = cake_backend.ROUTER_RULES_BY_SM[("sm_107a", 212)]
    key = ("sm_107a", "kk", 5120, 256)
    monkeypatch.setitem(own, key, "S2")
    assert cake_backend._router_memo().forward_verdicts
    own[key] = None
    assert not cake_backend._router_memo().forward_verdicts
    # a change of a weight-gradient row renews the memo too (its conclusions may differ)
    memo = cake_backend._router_memo()
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_100a", "nn_t", 4096, 256), "pairs"
    )
    renewed = cake_backend._router_memo()
    assert renewed is not memo and renewed.tracked and not renewed.forward_verdicts
    # a plain table bound in place of one: read on every call, so a change of it is seen without a generation
    plain = dict(cake_backend.ROUTER_RULES_BY_SM)
    monkeypatch.setattr(cake_backend, "ROUTER_RULES_BY_SM", plain)
    assert not cake_backend._router_memo().tracked
    assert not cake_backend._router_memo().forward_verdicts
    plain[("sm_107a", 200)] = {key: "S2"}
    assert cake_backend._router_memo().forward_verdicts
    del plain[("sm_107a", 200)]
    assert not cake_backend._router_memo().forward_verdicts
    # a counting table bound in place of one: kept again, and its changes are seen
    rules = cake_backend._VerdictTable(cake_backend.ROUTER_RULES)
    monkeypatch.setattr(cake_backend, "ROUTER_RULES", rules)
    monkeypatch.setattr(
        cake_backend,
        "ROUTER_RULES_BY_SM",
        cake_backend._VerdictTable(
            {k: cake_backend._VerdictTable(v) for k, v in plain.items()}
        ),
    )
    memo = cake_backend._router_memo()
    assert memo.tracked and memo.rules is rules and cake_backend._router_memo() is memo
    rules[("sm_100a", "kk", 4096, 256)] = "forward"
    assert cake_backend._router_memo().forward_verdicts
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    assert cake_backend._router_memo() is not memo


def _fake_device_facts(monkeypatch, parts):
    """``deterministic_backend().device_facts(index)`` of fake CUDA devices ``index -> (arch, sm_count)`` (CPU
    tests of the convenience path's memo, which keys its conclusions by device index)."""
    facts = {
        i: det_backend.DeviceFacts(index=i, arch=a, sm_count=n, l2_bytes=1 << 27)
        for i, (a, n) in parts.items()
    }
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts[device])


def test_k2_memo_is_the_planners_check(monkeypatch):
    """``_k2_runs_without_verdict`` (the convenience path's check of a call without a split count, kept per device
    index and view geometry) is ``_k2_row_without_verdict`` on that device's architecture and SM count for every
    binding -- each device its own -- and follows the router tables: a verdict set at run time in the table of a
    device's SM count is seen at the next call (the row is planned), and so is its removal, without clearing any
    cache; the memo starts over when full and still answers the same."""
    parts = {0: ("sm_103a", 152), 1: ("sm_107a", 212), 2: ("sm_107a", 216)}
    _fake_device_facts(monkeypatch, parts)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    runs = cake_backend._k2_runs_without_verdict
    cases = []
    for hidden in (2048, 4096, 5120, 6144):
        for T in (64, 128, 192, 193, 224, 1024, 16231):
            X, W, G = (
                torch.empty(T, hidden),
                torch.empty(256, hidden),
                torch.empty(T, 256),
            )
            cases += [
                (X, W.t(), torch.empty(T, 256)),
                (G.t(), X, torch.empty(256, hidden)),
                (G, W, torch.empty(T, hidden)),  # the kn class: left to the planner
                (X.bfloat16(), W.t(), torch.empty(T, 256)),  # refused by the planner
            ]

    def check():
        n = 0
        for index, (arch, sm_count) in parts.items():
            for A, B, out in cases:
                want = cake_backend._k2_row_without_verdict(A, B, out, arch, sm_count)
                got = runs(cake_backend._k2_geometry(A, B, out), index)
                assert got == want, (index, tuple(A.shape), A.stride(), A.dtype)
                n += want
        return n

    kept = check()
    assert kept and check() == kept  # the second pass reads the memo
    memo = cake_backend._router_memo()
    assert len(memo.without_verdict) == len(parts) * len(cases)
    # a verdict set at run time in the table of the 212-SM part's SM count: seen at the next call
    G, X = torch.empty(1000, 256), torch.empty(1000, 6144)
    geometry = cake_backend._k2_geometry(G.t(), X, torch.empty(256, 6144))
    assert runs(geometry, 1) and runs(geometry, 0)
    own = cake_backend.ROUTER_RULES_BY_SM[("sm_107a", 212)]
    key = ("sm_107a", "nn_t", 6144, 256)
    # the 212-SM hidden-6144 windows (SIMT to 224 tokens, head/tail 16172 to 16384 tokens)
    shipped = own[key]
    monkeypatch.setitem(own, key, ((1, 1024, "pairs"),))
    assert not runs(geometry, 1) and runs(geometry, 0) and runs(geometry, 2)
    own[key] = shipped
    assert runs(geometry, 1)
    assert check() == kept
    # a full memo starts over (from an empty memo: every check is a miss)
    monkeypatch.setattr(cake_backend, "_ROUTER_MEMO_SIZE", 5)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    assert check() == kept
    assert 0 < len(cake_backend._router_memo().without_verdict) <= 5


def test_k2_memo_under_concurrent_table_changes(monkeypatch):
    """Threads checking router calls while another thread sets and removes a verdict: no call fails, and once the
    tables stop changing every conclusion is the planner's for the final tables (a conclusion drawn while a table
    changed is not kept past that change)."""
    _fake_device_facts(monkeypatch, {0: ("sm_103a", 152)})
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    geometries = []
    for T in (64, 128, 192, 224, 1024):
        G, X = torch.empty(T, 256), torch.empty(T, 4096)
        geometries.append(
            (cake_backend._k2_geometry(G.t(), X, torch.empty(256, 4096)), T)
        )
    key = ("sm_103a", "nn_t", 4096, 256)
    monkeypatch.setitem(cake_backend.ROUTER_RULES, key, None)
    stop, errors = threading.Event(), []

    def reader():
        try:
            while not stop.is_set():
                for geometry, _ in geometries:
                    cake_backend._k2_runs_without_verdict(geometry, 0)
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(200):
        cake_backend.ROUTER_RULES[key] = ((1, 128, "pairs"),) if i % 2 else None
    cake_backend.ROUTER_RULES[key] = ((1, 128, "pairs"),)
    stop.set()
    for t in threads:
        t.join()
    assert not errors
    for geometry, T in geometries:
        assert cake_backend._k2_runs_without_verdict(geometry, 0) == (T > 128), T


# (arch, SM count) of the measured router tables: ROUTER_RULES on 152-SM GB200 / GB300 and 216-SM R200 parts,
# ROUTER_RULES_BY_SM on 148-SM B200 / B300 and 208-SM and 212-SM R200 parts
ROUTER_TABLE_PARTS = (
    ("sm_100a", 148),
    ("sm_100a", 152),
    ("sm_103a", 148),
    ("sm_103a", 152),
    ("sm_107a", 208),
    ("sm_107a", 212),
    ("sm_107a", 216),
)


def test_router_rule_spec_reads_the_table_of_the_sm_count(monkeypatch):
    """The convenience path's no-verdict check reads the router table of the device's SM count (``router_rules``):
    ``ROUTER_RULES`` on 152 SMs (sm_100a, sm_103a) and 216 SMs (sm_107a), the part's own table of
    ``ROUTER_RULES_BY_SM`` on 148 SMs (sm_100a, sm_103a) and 208 and 212 SMs (sm_107a).  At every row of that table
    and the edges of its windows the spec is the table's, ``None`` exactly where the device's planner keeps the row on
    the deterministic program for want of a verdict; a verdict in only one table is seen on exactly its SM count."""
    _register(
        monkeypatch,
        "sm_107a",
        "router_fp32_gemm_atomic_kk",
        "router_fp32_gemm_atomic_nn_t",
        "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX,
        cake_backend.router_head_tail_template("sm_107a", 212),
    )
    checked = 0
    for arch, sms in ROUTER_TABLE_PARTS:
        table, holds = cake_backend.router_rules(arch, sms)
        own = cake_backend.ROUTER_RULES_BY_SM.get((arch, sms))
        assert holds and table is (cake_backend.ROUTER_RULES if own is None else own)
        assert (own is None) == (sms == cake_backend.ROUTER_SM_COUNT[arch])
        for (key_arch, layout, hidden, experts), rule in table.items():
            if key_arch != arch:
                continue
            for T in sorted(_router_spec_tokens(rule)):
                M, N, K = _router_launch_dims(layout, hidden, experts, T)
                want = rule
                if isinstance(rule, tuple):
                    want = next((s for lo, hi, s in rule if lo <= T <= hi), None)
                spec = cake_backend._router_rule_spec(
                    layout, arch=arch, sm_count=sms, M=M, N=N, K=K
                )
                assert spec == want, (arch, sms, layout, hidden, T)
                part = cake_backend.router_partition(
                    layout, arch=arch, M=M, N=N, K=K, sm_count=sms
                )
                if spec is None:
                    assert part is None, (arch, sms, layout, hidden, T)
                checked += 1
    assert checked > 300
    # the sm_107a hidden-5120 weight gradient: "det_s12" at 16172 to 16384 tokens on 212 SMs only (no window on 208
    # and 216 SMs); hidden 6144: "ksum_S6" at 16172 to 16231 tokens on 212 SMs only
    for sms in (208, 212, 216):
        for T in (16171, 16172, 16231, 16384, 16385):
            want = "det_s12" if sms == 212 and 16172 <= T <= 16384 else None
            assert (
                cake_backend._router_rule_spec(
                    "nn_t", arch="sm_107a", sm_count=sms, M=5120, N=256, K=T
                )
                == want
            ), (sms, T)
            want = "ksum_S6" if sms == 212 and 16172 <= T <= 16231 else None
            assert (
                cake_backend._router_rule_spec(
                    "nn_t", arch="sm_107a", sm_count=sms, M=6144, N=256, K=T
                )
                == want
            ), (sms, T)
    # a head/tail verdict in the 212-SM table only, then in ROUTER_RULES only (216 SMs): the no-verdict check of a
    # binding of the row follows the table of each SM count
    key = ("sm_107a", "nn_t", 6144, 256)
    tables = {
        part: dict(rows) for part, rows in cake_backend.ROUTER_RULES_BY_SM.items()
    }
    tables[("sm_107a", 212)][key] = ((16172, 16384, "H0.5x4"),)
    monkeypatch.setattr(cake_backend, "ROUTER_RULES_BY_SM", tables)
    G, X = torch.empty(16231, 256), torch.empty(16231, 6144)
    out = torch.empty(256, 6144)
    keeps = {
        sms: cake_backend._k2_row_without_verdict(G.t(), X, out, "sm_107a", sms)
        for sms in (208, 212, 216)
    }
    assert keeps == {208: True, 212: False, 216: True}
    plan, _ = cake_backend.plan_atomic_router(
        G.t(), X, out, arch="sm_107a", sm_count=212
    )
    assert plan is not None and plan.head_spans > 0
    monkeypatch.setitem(cake_backend.ROUTER_RULES, key, ((16172, 16384, "H0.5x4"),))
    tables[("sm_107a", 212)][key] = None
    keeps = {
        sms: cake_backend._k2_row_without_verdict(G.t(), X, out, "sm_107a", sms)
        for sms in (208, 212, 216)
    }
    assert keeps == {208: True, 212: True, 216: False}


@pytest.mark.parametrize("splits", [None, 2])
def test_k2_key_is_the_convenience_key(splits):
    """The cached K2 launch reads the call's geometry once (``_k2_geometry``) for its checks and its key; the key is
    the deterministic package's ``convenience_key`` of ``("k2", splits)`` and the three views."""
    base = torch.empty(4096 + 8)
    for A, B, out in (
        (torch.empty(64, 512), torch.empty(256, 512).t(), torch.empty(64, 256)),
        (
            torch.empty(64, 256).t(),
            torch.empty(64, 512),
            torch.empty(256, 520)[:, :512],
        ),
        (
            base[1:4097].view(64, 64),
            base[8:4104].view(64, 64),
            torch.empty(64, 64, dtype=torch.bfloat16),
        ),
        (torch.empty(2, 8, 8), torch.empty(8, 8), torch.empty(8)),
    ):
        geometry = cake_backend._k2_geometry(A, B, out)
        assert cake_backend._k2_key(
            3, 77, splits, A, B, out, geometry, out.data_ptr()
        ) == det_backend.convenience_key(3, 77, ("k2", splits), A, B, out)


def _sm103a_wgrad_stream_k(monkeypatch) -> None:
    """The sm_103a hidden-5120 weight-gradient row with a two-unit stream-K window ``"S2"`` from 1 to 192 tokens in
    place of its SIMT and "waves" windows and with its former head/tail window ``"H0.5x3"`` at 16172 to 32768 tokens
    (for these tests only): the binding tests take its partition at 192 tokens on a 152-SM part (two partials per
    element; the shipped "waves" window gives more there and runs the deterministic program) and bind the registered
    head/tail program at 16172 tokens (no shipped table holds a head/tail window)."""
    key = ("sm_103a", "nn_t", 5120, 256)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES,
        key,
        ((1, 192, "S2"), (16172, 32768, "H0.5x3")),
    )


def _same_arguments(got: list, want: list) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if isinstance(w, torch.Tensor):
            assert isinstance(g, torch.Tensor) and _geometry(g) == _geometry(w), i
        else:
            assert type(g) is type(w) and g == w, (i, g, w)


@pytest.mark.parametrize(
    "case",
    [
        "wgrad_stream_k",
        "wgrad_head_tail",
        "fwd_two_splits",
        "padded_out",
        "torch_chain",
    ],
)
def test_k2_bound_arguments_are_the_ordered_arguments(monkeypatch, case):
    """A cached K2 launch orders its arguments once (``_k2_binding``) and binds the arena's BF16x3 stack once per arena
    growth: with a call's views in place its kernel and fused-split argument lists are the lists the launch orders
    per call from ``_k2_values`` / ``_split_values`` (the same tensors' geometry, the same values in the same
    positions), ``out`` stands for both of its arguments, and the fill geometry is ``_out_fill``'s.  A grown arena is
    carved again."""
    monkeypatch.setattr(cake_backend.tvm_ffi, "from_dlpack", lambda t, **_: t)
    monkeypatch.setattr(cake_backend, "_CU_DRIVER", False)
    # 192 tokens at hidden 5120: the two-unit stream-K window set for these tests
    _sm103a_wgrad_stream_k(monkeypatch)
    arch, sm_count, T, H, splits = "sm_103a", 152, 192, 5120, None
    if case == "wgrad_head_tail":
        T = 16172
    elif case == "fwd_two_splits":
        splits = 2
    X, W = torch.empty(T, H), torch.empty(256, H)
    G = (
        torch.empty(T, 256 + 16)[:, :256]
        if case == "torch_chain"
        else torch.empty(T, 256)
    )
    if case == "fwd_two_splits":
        A, B, out = X, W.t(), torch.empty(T, 256)
    else:
        out = (
            torch.empty(256, H + 64)[:, :H]
            if case == "padded_out"
            else torch.empty(256, H)
        )
        A, B = G.t(), X
    plan, reason = cake_backend.plan_atomic_router(
        A, B, out, arch=arch, sm_count=sm_count, splits=splits
    )
    assert plan is not None, reason
    assert (plan.head_spans > 0) == (case == "wgrad_head_tail")
    module_name = cake_jit.select_module(arch, plan.template)
    arg_plan = cake_jit.arg_plan_of(cake_jit.MODULES[module_name])
    a_view, split_src = cake_backend._k2_views(A, B, plan)
    split_module = det_backend._fused_split_for(arch, split_src)
    assert (split_module is None) == (case == "torch_chain")
    split_arg_plan = (
        None
        if split_module is None
        else det_backend.arg_plan_of(det_backend.MODULES[split_module])
    )
    bound = cake_backend._k2_binding(
        plan, out, split_src, module_name, arg_plan, split_module, split_arg_plan, 0
    )
    n = split_src.numel()
    arena = cake_backend._AtomicArena(torch.device("cpu"), (0, 0))
    for grow in (1, 4):
        arena.require("k2_stack", 3 * n * grow)
        if split_module is None:
            arena.require("k2_split_scratch", n * grow)
        assert bound.version != arena.version
        bound.carve(arena, split_module is None)
        assert bound.version == arena.version
        parts = arena.buffers["k2_stack"][: 3 * n].view(3, *split_src.shape)
        assert _geometry(bound.parts) == _geometry(parts)
        if split_module is None:
            resid = arena.buffers["k2_split_scratch"][:n].view(split_src.shape)
            assert _geometry(bound.resid) == _geometry(resid)
        else:
            assert bound.resid is None
        out32 = out if bound.out_contiguous else det_backend.flat_alias(out)
        assert bound.out_contiguous == (case != "padded_out")
        kernel_args, split_args = bound.arguments((a_view, out32, out, split_src))
        _same_arguments(
            kernel_args,
            cake_backend._order_atomic_arguments(
                module_name,
                arg_plan,
                cake_backend._k2_values(plan, A=a_view, parts=parts, out=out),
                plan.grid,
            ),
        )
        if split_module is None:
            assert split_args is None
        else:
            _same_arguments(
                split_args,
                cake_backend._order_atomic_arguments(
                    split_module,
                    split_arg_plan,
                    det_backend._split_values(split_src, parts, n),
                    det_backend._split_grid(n),
                ),
            )
    if plan.part_tiles:
        fill = cake_backend._out_fill(out, 0)
        assert bound.fill == (fill.rows, fill.cols, fill.pitch)
    else:
        assert bound.fill is None


def _cpu_k2_launches(monkeypatch, arch="sm_103a"):
    """CPU binding tests of the router entry points' K2 launches (:func:`_cpu_k1_launches`, nothing is loaded or
    launched): the entry points' tensors taken as on device 0, the zero fill of ``out`` (part tiles) recorded, fresh
    K2 caches and router-table memo; returns the stand-in module's entry and the fill recorder."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    monkeypatch.setattr(cake_backend, "_k2_index", lambda A, B, out: 0)
    monkeypatch.setattr(cake_backend, "_CU_DRIVER", False)
    fill = Recorder()
    monkeypatch.setattr(cake_backend, "_fill_rows", fill)
    for cache in ("_K2_ATOMIC_CACHE", "_K2_SPLITK_CACHE", "_K2_ENTRY_CACHE"):
        monkeypatch.setattr(cake_backend, cache, type(getattr(cake_backend, cache))())
    memo = cake_backend._RouterMemo(-1, None, None, False, True)
    monkeypatch.setattr(cake_backend, "_ROUTER_MEMO", memo)
    return entry_fn, fill


# A router entry point's call on a 152-SM sm_103a part: (op, T, hidden, splits, padded out, padded G, whether it runs
# this package's program -- else the deterministic program through this package's binding).  192 tokens at hidden
# 5120: past the SIMT window of the input gradient (to 128 tokens); the weight gradient in the two-unit stream-K window
# set for these tests (:func:`_sm103a_wgrad_stream_k`).  The forward without a split count at 224 tokens: past its
# SIMT windows (to 192 tokens; :func:`test_router_forward_entry_binds_the_simt_program`)
# fmt: off
ROUTER_ENTRY_CASES = {
    "fwd": ("fwd", 224, 5120, None, False, False, False),
    "fwd_two_splits": ("fwd", 192, 5120, 2, False, False, True),
    "fwd_padded_out": ("fwd", 224, 5120, None, True, False, False),
    "dgrad": ("dgrad", 192, 5120, None, False, False, False),
    "dgrad_padded_out": ("dgrad", 192, 5120, None, True, False, False),
    "wgrad_stream_k": ("wgrad", 192, 5120, None, False, False, True),
    "wgrad_head_tail": ("wgrad", 16172, 5120, None, False, False, True),
    "wgrad_no_verdict": ("wgrad", 1000, 6144, None, False, False, False),
    "wgrad_padded_out": ("wgrad", 192, 5120, 2, True, False, True),
    "wgrad_torch_chain": ("wgrad", 192, 5120, None, False, True, True),
}
# fmt: on


def _router_entry_call(case):
    """``(call, out)``: a router entry point's call of ``case`` (:data:`ROUTER_ENTRY_CASES`)
    on fresh CPU tensors, and its output."""
    op, T, H, splits, pad_out, pad_g, _ = ROUTER_ENTRY_CASES[case]
    X, W = torch.empty(T, H), torch.empty(256, H)
    G = torch.empty(T, 256 + 16)[:, :256] if pad_g else torch.empty(T, 256)
    rows, cols = {"fwd": (T, 256), "dgrad": (T, H), "wgrad": (256, H)}[op]
    out = torch.empty(rows, cols + 64)[:, :cols] if pad_out else torch.empty(rows, cols)
    kw = dict() if op == "dgrad" else dict(splits=splits)
    call = {
        "fwd": lambda: cake_backend.cake_dense_gemm_nondet_train_router_forward(
            X, W, out, **kw
        ),
        "dgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router_dgrad(
            G, W, out, **kw
        ),
        "wgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
            G, X, out, **kw
        ),
    }[op]
    return call, out


@pytest.mark.parametrize("case", sorted(ROUTER_ENTRY_CASES))
def test_router_entry_points_launch_their_paths_launches(monkeypatch, case):
    """A router entry point's call goes through ``_K2_ENTRY_CACHE``: planned once per key
    of its own tensors into the cached launch its own path takes (the same ``_K2_SPLITK_CACHE`` / ``_K2_ATOMIC_CACHE``
    entry, under the same key), it issues that path's launches -- the same programs with the same arguments, the
    same split and zero fill -- and the next calls go from the key to the bound launch (no geometry, layout, key of
    the launch or planning)."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch)
    _sm103a_wgrad_stream_k(monkeypatch)
    split = Recorder()  # the torch chain's split
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    op, *_, atomic = ROUTER_ENTRY_CASES[case]
    call, out = _router_entry_call(case)
    with monkeypatch.context() as m:
        m.setattr(cake_backend, "_k2_entry", lambda *args: None)
        assert call() is out
    recorders = (entry_fn, fill, split)
    want = [list(r.calls) for r in recorders]
    # the deterministic program's split-K partials (4 forward, 6 weight gradient) are summed in the chain
    sums = not atomic and op != "dgrad"
    assert len(want[0]) == (1 if case.endswith("torch_chain") else 2 + sums)
    assert len(want[2]) == (1 if case.endswith("torch_chain") else 0)
    inner = cake_backend._K2_ATOMIC_CACHE if atomic else cake_backend._K2_SPLITK_CACHE
    ((inner_key, launch),) = [(k, e) for k, e in inner.items() if e.plan is not None]
    assert atomic or (launch.sum_entry is not None) == sums
    for again in range(3):
        for r in recorders:
            r.calls.clear()
        assert call() is out
        for r, expected in zip(recorders, want, strict=True):
            assert len(r.calls) == len(expected)
            for (args, kwargs), (w_args, w_kwargs) in zip(
                r.calls, expected, strict=True
            ):
                _same_arguments(list(args), list(w_args))
                assert kwargs == w_kwargs
        ((key, entry),) = cake_backend._K2_ENTRY_CACHE.items()
        assert key[2][0] == op and key[2][2] == 1  # the PDL flag: the device's default
        assert entry.launch is launch and entry.atomic == atomic
        assert entry.arena is launch.arena
        assert inner[inner_key] is launch
        if again == 0:
            for name in ("_k2_geometry", "_k2_layout", "_k2_key", "_plan_k2_entry"):
                monkeypatch.setattr(
                    cake_backend,
                    name,
                    lambda *args, _n=name: pytest.fail(f"{_n} on a cached call"),
                )
    assert len(fill.calls) == int(atomic and launch.k2.fill is not None)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
@pytest.mark.parametrize("T", [1, 8, 9, 16, 17, 128, 129, 192, 193])
def test_router_forward_entry_binds_the_simt_program(monkeypatch, arch, T):
    """Without a split count the forward's call in a SIMT window of the architecture's part (:data:`ROUTER_FWD_PARTS`:
    152 SMs, 212 SMs on sm_107a) binds the cluster program's launch in ``_K2_ENTRY_CACHE`` (planned once into
    ``_K2_ATOMIC_CACHE``): one launch per call of the program with the entry point's tensors -- ``X``, ``W`` and
    ``out`` read in place --, the launch's dimensions and row strides, the device's PDL flag and the grid of the
    cluster tiles; no split, no fill.  Past the windows the call binds the deterministic program (four splits) through
    this package's binding."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch, arch)
    sms = ROUTER_FWD_PARTS[arch]
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sms, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    split = Recorder()
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    H, E = 5120, 256
    spec = next((s for a, b, s in _fwd_windows(arch, sms, H) if a <= T <= b), None)
    # in a window the activations' rows are strided (read in place)
    X = torch.empty(T, H) if spec is None else torch.empty(T, H + 8)[:, :H]
    W, out = torch.empty(E, H), torch.empty(T, E)
    for _ in range(3):
        assert (
            cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, out) is out
        )
    ((key, entry),) = cake_backend._K2_ENTRY_CACHE.items()
    assert key[2][0] == "fwd" and key[2][1] is None and key[2][2] == 1
    assert not fill.calls and entry.launch is not None
    if spec is None:
        assert not entry.atomic and not cake_backend._K2_ATOMIC_CACHE
        assert entry.launch.plan.splits == 4
        return
    tile = cake_backend.router_forward_simt_tile(spec)
    assert not split.calls
    assert entry.atomic and entry.launch.plan.simt == tile
    assert entry.launch.plan.template == cake_backend.router_forward_simt_template(tile)
    assert entry.launch.plan.grid == (8 * E // tile[2], -(-T // tile[3]), 1)
    ((_, launch),) = cake_backend._K2_ATOMIC_CACHE.items()
    assert entry.launch is launch and not cake_backend._K2_SPLITK_CACHE
    assert len(entry_fn.calls) == 3
    # X, W, out, then T, N, K, ldx, ldw, ldo, the PDL flag and the grid
    want = [T, E, H, H + 8, H, E, 1, 8 * E // tile[2], -(-T // tile[3]), 1]
    for args, kwargs in entry_fn.calls:
        assert not kwargs and list(args[3:]) == want, args[3:]
        assert args[0].data_ptr() == X.data_ptr() and args[1].data_ptr() == W.data_ptr()
        assert args[2].data_ptr() == out.data_ptr()


@pytest.mark.parametrize("T", [1, 17, 64, 128, 129])
def test_router_dgrad_entry_binds_the_simt_program(monkeypatch, T):
    """The input gradient's call in a SIMT window of a 152-SM sm_100a part binds the SIMT
    program's launch in ``_K2_ENTRY_CACHE`` (planned once into ``_K2_ATOMIC_CACHE``): one launch per call of the
    program with the entry point's tensors -- ``G[None]``, ``W`` and ``out`` read in place --, the launch's
    dimensions and row strides, no PDL flag and the grid of ``out``'s tiles; no split, no fill.  One token past the
    window the call binds the deterministic program (one split) through this package's binding."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch, "sm_100a")
    split = Recorder()
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    H, E = 5120, 256
    G, W = torch.empty(T, E), torch.empty(E, H)
    out = torch.empty(T, H + 64)[:, :H]
    for _ in range(3):
        assert cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out) is out
    ((key, entry),) = cake_backend._K2_ENTRY_CACHE.items()
    assert key[2][0] == "dgrad" and key[2][1] is None
    assert not fill.calls and not split.calls and entry.launch is not None
    if T > 128:
        assert not entry.atomic and not cake_backend._K2_ATOMIC_CACHE
        assert entry.launch.plan.splits == 1
        return
    assert entry.atomic and entry.launch.plan.simt == (64, 32, 4)
    assert entry.launch.plan.template == "router_fp32_dgrad_simt_m32n64s4"
    ((_, launch),) = cake_backend._K2_ATOMIC_CACHE.items()
    assert entry.launch is launch and not cake_backend._K2_SPLITK_CACHE
    assert len(entry_fn.calls) == 3
    want = [G.unsqueeze(0), W, out, T, H, E, E, H, H + 64, 0, H // 64, -(-T // 32), 1]
    for args, kwargs in entry_fn.calls:
        assert not kwargs
        _same_arguments(list(args), want)
        assert args[0].data_ptr() == G.data_ptr() and args[1] is W
        assert args[2].data_ptr() == out.data_ptr()


class _ReportsCuda(torch.Tensor):
    """A CPU tensor whose ``is_cuda`` is true: the deterministic router planner reads the device's facts (for its
    ``router_splits``) only for an ``out`` on a CUDA device."""

    @property
    def is_cuda(self):
        return True


# The parts with weight-gradient rows in the deterministic package's ROUTER_SPLIT_RULES, (arch, SM count)
ROUTER_SPLIT_RULE_PARTS = sorted(
    {
        (a, sms)
        for a, sms, layout, _ in det_backend.ROUTER_SPLIT_RULES
        if layout == "nn_t"
    }
)


@pytest.mark.parametrize("arch,sm_count", ROUTER_SPLIT_RULE_PARTS)
def test_router_wgrad_split_rule_rows_take_the_deterministic_count(
    monkeypatch, arch, sm_count
):
    """The weight gradient on a part with rows in the deterministic package's ``ROUTER_SPLIT_RULES``, at the edges of
    each window, one token outside them and at the edges of this package's windows of the row among them: a token
    count where this package has neither a partition nor a SIMT window (:func:`router_partition`,
    :func:`router_simt`) runs the deterministic program through this package's binding at the deterministic
    package's count of the row (its ``router_splits``: the rule's count inside the window, the default one outside),
    or at the count of this package's ``"det_s<n>"`` window where one holds (:func:`router_det_splits`; hidden 6144:
    six splits, where the deterministic package's rule runs three), its partials reduced by this package's partial
    sum where it serves the count, else by ``torch.sum``; every other
    token count runs this package's program.  ``cake_dense_gemm_nondet_train_router_plan`` plans the same: the
    deterministic package's plan (its count) where the call runs the deterministic program."""
    _cpu_k2_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sm_count, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    det_launch, own_launch = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_launch_splitk_k2_binding", det_launch)
    monkeypatch.setattr(cake_backend, "_launch_atomic_k2_binding", own_launch)
    table, _ = cake_backend.router_rules(arch, sm_count)
    default = det_backend.ROUTER_SPLITS["nn_t"]
    ran = set()  # (hidden, tokens, splits) run on the deterministic program
    for (a, sms, layout, hidden), windows in det_backend.ROUTER_SPLIT_RULES.items():
        if (a, sms, layout) != (arch, sm_count, "nn_t"):
            continue
        rule = table.get((arch, "nn_t", hidden, 256))
        own = rule if isinstance(rule, tuple) else ()
        for lo, hi, splits in windows:
            tokens = {lo - 1, lo, hi, hi + 1} | {
                t
                for first, last, _ in own
                for t in (first - 1, first, last, last + 1)
                if lo - 1 <= t <= hi + 1
            }
            for T in sorted(tokens):
                want = splits if lo <= T <= hi else default
                assert (
                    det_backend.router_splits("nn_t", hidden, 256, T, arch, sm_count)
                    == want
                )
                kw = dict(arch=arch, sm_count=sm_count, M=hidden, N=256, K=T)
                runs_own = (
                    cake_backend.router_partition("nn_t", **kw) is not None
                    or cake_backend.router_simt("nn_t", **kw) is not None
                )
                X, G = torch.empty(T, hidden), torch.empty(T, 256)
                out = torch.empty(256, hidden).as_subclass(_ReportsCuda)
                plan, *_ = det_backend.cake_dense_gemm_det_train_router_plan(
                    G.t(), X, out
                )
                assert plan.splits == want, (hidden, T)
                plan = cake_backend.cake_dense_gemm_nondet_train_router_plan(
                    G.t(), X, out
                )
                assert (
                    isinstance(plan, cake_backend.CakeDenseGemmNondetRouterPlan)
                    == runs_own
                )
                cake_backend._drop_convenience_entries()
                det_launch.calls.clear()
                own_launch.calls.clear()
                assert (
                    cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, out)
                    is out
                )
                if runs_own:
                    assert len(own_launch.calls) == 1 and not det_launch.calls
                    continue
                own_n = cake_backend.router_det_splits("nn_t", **kw) or want
                assert plan[0].splits == own_n, (hidden, T)
                assert len(det_launch.calls) == 1 and not own_launch.calls
                (entry,) = cake_backend._K2_SPLITK_CACHE.values()
                assert entry.plan.splits == own_n, (hidden, T)
                served = own_n in cake_backend.PARTIALS_SUM_SPLITS
                assert (entry.sum_entry is not None) == served, (hidden, T)
                # a "det_s<n>" row keeps its split count in a planless atomic entry; every other row has none
                atomic_entries = list(cake_backend._K2_ATOMIC_CACHE.values())
                if own_n != want:
                    assert [(e.plan, e.det_splits) for e in atomic_entries] == [
                        (None, own_n)
                    ], (hidden, T)
                else:
                    assert all(e.plan is None for e in atomic_entries), (hidden, T)
                ran.add((hidden, T, own_n))
    # a rule's count ran on the deterministic program inside its window
    assert any(want != default for _, _, want in ran), sorted(ran)


def test_router_entry_cache_keys_and_routes_again(monkeypatch):
    """``_K2_ENTRY_CACHE`` keys a call by the device, the raw stream, the operation, the split count, the PDL flag and
    the entry point's tensors, and routes a key again after a router table changes (a forward verdict set at run time
    runs this package's program, its removal the deterministic one) or a ``ROUTES`` value does; a call whose tensors
    are not on one CUDA device, or made while a router table is a plain ``dict``, takes the entry point's own path.
    ``cake_dense_gemm_nondet_train_convenience_cache_clear`` drops the entries."""
    arch, sm_count = "sm_103a", 152
    X, W, out = torch.empty(64, 5120), torch.empty(256, 5120), torch.empty(64, 256)
    assert cake_backend._k2_entry("fwd", X, W, out, None, None) is None  # CPU tensors
    entry_fn, _ = _cpu_k2_launches(monkeypatch, arch)
    cache = cake_backend._K2_ENTRY_CACHE

    def fwd(**kw):
        n = len(entry_fn.calls)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, out, **kw)
            is out
        )
        assert len(entry_fn.calls) > n
        return cache[next(reversed(cache))]

    forward_verdicts = cake_backend._router_memo().forward_verdicts
    fwd()
    fwd(enable_pdl=False)
    fwd(splits=2)
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 7)
    fwd()
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 0)
    keys = list(cache)
    assert [k[:3] for k in keys] == [
        (0, 0, ("fwd", None, 1)),
        (0, 0, ("fwd", None, 0)),
        (0, 0, ("fwd", 2, 1)),
        (0, 7, ("fwd", None, 1)),
    ]
    assert [k[3:] for k in keys] == [keys[0][3:]] * 4
    assert cache[keys[2]].atomic and (cache[keys[2]].layout, cache[keys[2]].route) == (
        "k2_kk",
        "atomic",
    )
    assert [cache[k].arena.key for k in keys] == [(0, 0), (0, 0), (0, 0), (0, 7)]
    if not forward_verdicts:  # the forward without a split count reads no route
        assert all(cache[k].layout is None and not cache[k].atomic for k in keys[:2])
    # a forward verdict set at run time, and removed: this package's program, then the deterministic one
    table, _ = cake_backend.router_rules(arch, sm_count)
    if table is cake_backend.ROUTER_RULES:
        monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
    X, W, out = torch.empty(1000, 6144), torch.empty(256, 6144), torch.empty(1000, 256)
    key = (arch, "kk", 6144, 256)
    # no verdict at 1000 tokens (the row's SIMT windows end at 192 tokens)
    assert all(last < 1000 for _, last, _ in table.get(key) or ())
    before = fwd()
    assert not before.atomic and before.launch is not None
    monkeypatch.setitem(table, key, "forward")
    after = fwd()
    assert after.atomic and after.memo is not before.memo and after.route == "atomic"
    assert after.launch.plan is not None and after.launch.k2 is not None
    table[key] = None
    assert not fwd().atomic
    # a ROUTES value changed at run time (192 tokens: the two-unit stream-K window set for this test at hidden
    # 5120)
    _sm103a_wgrad_stream_k(monkeypatch)
    X, G, out = torch.empty(192, 5120), torch.empty(192, 256), torch.empty(256, 5120)

    def wgrad():
        n = len(entry_fn.calls)
        assert cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, out) is out
        assert len(entry_fn.calls) > n
        return cache[next(reversed(cache))]

    assert wgrad().atomic
    monkeypatch.setitem(cake_backend.ROUTES, "k2_nn_t", "det")
    routed = wgrad()
    assert not routed.atomic and routed.launch is not None
    assert (routed.layout, routed.route) == ("k2_nn_t", "det")
    monkeypatch.setitem(cake_backend.ROUTES, "k2_nn_t", "atomic")
    assert wgrad().atomic
    # a router table rebound to a plain dict: every call takes the entry point's own path
    size = len(cache)
    with monkeypatch.context() as m:
        m.setattr(cake_backend, "ROUTER_RULES", dict(cake_backend.ROUTER_RULES))
        assert cake_backend._k2_entry("wgrad", G, X, out, None, None) is None
        n = len(entry_fn.calls)
        assert cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, out) is out
        assert len(entry_fn.calls) > n and len(cache) == size
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    assert not cache and not cake_backend._ATOMIC_ARENAS


# Router verdicts changed at run time, op -> (arch, SM count, tokens, hidden, first verdict, second verdict): the
# forward on 152-SM sm_100a (one unit per tile, then two), the input gradient on 212-SM sm_107a (the part's own table
# in ROUTER_RULES_BY_SM: the 32 x 64 SIMT tile in eight slices, then the 64 x 64 tile in four), the weight gradient on
# 152-SM sm_100a (the 64 x 64 SIMT tile, then the stream-K "pairs" partition)
# fmt: off
ROUTER_VERDICT_CHANGES = {
    "fwd": ("sm_100a", 152, 64, 5120, "S1", "S2"),
    "dgrad": ("sm_107a", 212, 64, 5120, ((1, 64, "simt_n64m32s8"),), ((1, 64, "simt_n64m64s4"),)),
    "wgrad": ("sm_100a", 152, 64, 5120, ((1, 256, "simt_n64m64s2"),), ((1, 256, "pairs"),)),
}
# fmt: on


@pytest.mark.parametrize("tracked", [True, False], ids=["verdict_tables", "dicts"])
@pytest.mark.parametrize("path", ["entry_point", "cake_dense_gemm_nondet_train_router"])
@pytest.mark.parametrize("op", sorted(ROUTER_VERDICT_CHANGES))
def test_router_verdict_changed_at_run_time_plans_the_launch_again(
    monkeypatch, op, path, tracked
):
    """A router verdict changed between two calls of the same tensors: the next call runs the launch the planner gives
    under the new verdict, not the cached launch of the old one -- through the router entry point (its
    ``_K2_ENTRY_CACHE`` entry and the ``_K2_ATOMIC_CACHE`` launch under it) and through ``cake_dense_gemm_nondet_train_router``
    (``_K2_ATOMIC_CACHE``); while no table changes, a call takes its cached launch.  With the router tables rebound to
    plain dicts (read on every call) a change is seen at the next call too.  A launch with an explicit split count
    reads no router table: it is kept across a change."""
    arch, sm_count, T, H, first, second = ROUTER_VERDICT_CHANGES[op]
    entry_fn, _ = _cpu_k2_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sm_count, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    if not tracked:
        monkeypatch.setattr(
            cake_backend, "ROUTER_RULES", dict(cake_backend.ROUTER_RULES)
        )
        monkeypatch.setattr(
            cake_backend,
            "ROUTER_RULES_BY_SM",
            {part: dict(t) for part, t in cake_backend.ROUTER_RULES_BY_SM.items()},
        )
    table, measured = cake_backend.router_rules(arch, sm_count)
    assert measured
    key = (arch, {"fwd": "kk", "dgrad": "kn", "wgrad": "nn_t"}[op], H, 256)
    X, W, G = torch.empty(T, H), torch.empty(256, H), torch.empty(T, 256)
    A, B, out = {
        "fwd": (X, W.t(), torch.empty(T, 256)),
        "dgrad": (G, W, torch.empty(T, H)),
        "wgrad": (G.t(), X, torch.empty(256, H)),
    }[op]
    entry_point = {
        "fwd": lambda: cake_backend.cake_dense_gemm_nondet_train_router_forward(
            X, W, out
        ),
        "dgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router_dgrad(
            G, W, out
        ),
        "wgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
            G, X, out
        ),
    }[op]

    def call():
        """The cached launch a call ran, after checking that it launched."""
        n = len(entry_fn.calls)
        if path == "cake_dense_gemm_nondet_train_router":
            assert cake_backend.cake_dense_gemm_nondet_train_router(A, B, out) is out
        else:
            assert entry_point() is out
        assert len(entry_fn.calls) > n
        ((_, launch),) = cake_backend._K2_ATOMIC_CACHE.items()
        if path == "entry_point" and tracked:
            ((_, entry),) = cake_backend._K2_ENTRY_CACHE.items()
            assert entry.atomic and entry.launch is launch
        else:  # cake_dense_gemm_nondet_train_router, or an entry point while a router table is a plain dict
            assert not cake_backend._K2_ENTRY_CACHE
        return launch

    plans = []
    for verdict in (first, second, first):
        monkeypatch.setitem(table, key, verdict)
        launch = call()
        want, _ = cake_backend.plan_atomic_router(
            A, B, out, arch=arch, sm_count=sm_count
        )
        assert want is not None and launch.plan == want, verdict
        again = call()
        # kept while no table changes; a plain dict is read on every call
        assert (again is launch) == tracked and again.plan == want
        plans.append(want)
    assert plans[0] != plans[1] and plans[2] == plans[0]
    if path == "cake_dense_gemm_nondet_train_router" and op != "dgrad":

        def with_splits():
            assert (
                cake_backend.cake_dense_gemm_nondet_train_router(A, B, out, splits=2)
                is out
            )
            (kept,) = [
                e for k, e in cake_backend._K2_ATOMIC_CACHE.items() if k[2][1][0] == 2
            ]
            return kept

        kept = with_splits()
        assert kept.memo is None and kept.plan is not None
        monkeypatch.setitem(table, key, second)
        monkeypatch.setattr(
            cake_backend, "_plan_atomic_k2", lambda *args: pytest.fail("planned again")
        )
        assert with_splits() is kept


def test_router_entry_without_a_bound_launch_takes_the_entry_points_path(monkeypatch):
    """A key whose launch is not bound, or whose plan does not take the entry point's tensors as its launch views,
    is kept without a launch: its calls take the entry point's own path (without planning the key again), and a plan
    with the entry point's flags is bound."""
    _cpu_k2_launches(monkeypatch)
    G, W, out = torch.empty(192, 256), torch.empty(256, 5120), torch.empty(192, 5120)
    arena = cake_backend._AtomicArena(torch.device("cpu"), (0, 0))
    memo = cake_backend._router_memo()

    def plan(flags, bound):
        swapped, a_mn, b_mn = flags
        return cake_backend._CachedSplitK2(
            plan=SimpleNamespace(swapped=swapped, a_mn=a_mn, b_mn=b_mn),
            entry=Recorder(),
            arena=arena,
            k2=bound,
        )

    dgrad_flags = cake_backend._DGRAD_VIEWS
    for launch, kept in (
        (plan(dgrad_flags, None), False),
        (plan(cake_backend._FORWARD_VIEWS, object()), False),
        (plan(dgrad_flags, object()), True),
    ):
        cake_backend._K2_SPLITK_CACHE.clear()
        monkeypatch.setattr(
            cake_backend, "_plan_splitk_k2", lambda *args, launch=launch: launch
        )
        entry = cake_backend._plan_k2_entry("dgrad", G, W, out, None, 0, 0, 1, memo)
        assert (entry.launch is launch) == kept and (entry.arena is arena) == kept
        # no input-gradient verdict at 192 tokens on a 152-SM sm_103a part: the deterministic program
        assert entry.memo is memo and entry.layout == "k2_kn" and not entry.atomic
    monkeypatch.setattr(
        cake_backend, "_plan_splitk_k2", lambda *args: plan(dgrad_flags, None)
    )
    cake_backend._K2_SPLITK_CACHE.clear()
    path = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", path)
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out) is path.result
    )
    ((key, entry),) = cake_backend._K2_ENTRY_CACHE.items()
    assert entry.launch is None and key[2] == ("dgrad", None, 1)
    monkeypatch.setattr(
        cake_backend, "_plan_k2_entry", lambda *args: pytest.fail("planned again")
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out) is path.result
    )
    assert len(path.calls) == 2


def test_router_entry_cache_eviction_keeps_the_arenas(monkeypatch):
    """An entry of ``_K2_ENTRY_CACHE`` uses its launch's arena, as the launch's own entry does; eviction (the cache
    bound) drops entries but no arena: an arena outlives its last cached entry with every buffer it holds (a CUDA
    graph may still replay a launch bound to them), a call on its stream plans into it again, and only
    ``cake_dense_gemm_nondet_train_convenience_cache_clear`` drops it."""
    _cpu_k2_launches(monkeypatch)
    monkeypatch.setattr(det_backend, "CONVENIENCE_CACHE_SIZE", 3)
    W = torch.empty(256, 6144)
    caches = (
        cake_backend._K2_ENTRY_CACHE,
        cake_backend._K2_SPLITK_CACHE,
        cake_backend._K2_ATOMIC_CACHE,
    )

    def dgrad(T):
        G, out = torch.empty(T, 256), torch.empty(T, 6144)
        assert cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out) is out
        arenas = cake_backend._ATOMIC_ARENAS
        entries = [e for c in caches for e in c.values()]
        assert all(arenas.get(e.arena.key) is e.arena for e in entries)
        return {k: sum(e.arena is a for e in entries) for k, a in arenas.items()}

    def held(arena):
        return {id(b) for b in (*arena.buffers.values(), *arena.retired)}

    # past the hidden 6144 SIMT window of the input gradient (to 64 tokens): the deterministic program's binding
    for T in (128, 192, 256, 320, 384):
        users = dgrad(T)
    assert len(caches[0]) == len(caches[1]) == 3 and users == {(0, 0): 6}
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 7)
    assert dgrad(128) == {(0, 0): 4, (0, 7): 2}  # an arena of the stream's own
    arena7 = cake_backend._ATOMIC_ARENAS[(0, 7)]
    buffers7 = held(arena7)
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 0)
    for T in (128, 192, 256):
        users = dgrad(T)
    # evicted from both caches: the arena of stream 7 stays, with its buffers
    assert users == {(0, 0): 6, (0, 7): 0}
    assert cake_backend._ATOMIC_ARENAS[(0, 7)] is arena7 and held(arena7) == buffers7
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 7)
    assert dgrad(128) == {(0, 0): 4, (0, 7): 2}  # planned into the same arena again
    assert cake_backend._ATOMIC_ARENAS[(0, 7)] is arena7 and held(arena7) == buffers7
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    assert not cake_backend._ATOMIC_ARENAS and not any(caches)


def test_deterministic_backend_follows_the_binding_and_sys_modules(monkeypatch):
    """``deterministic_backend`` returns the module an import of the bound package's ``cake_backend`` returns: a
    rebinding and a replaced ``sys.modules`` entry are seen on the next call."""
    assert cake_backend.deterministic_backend() is det_backend
    fake_pkg, fake = (
        ModuleType("vendored_det_pkg2"),
        ModuleType("vendored_det_pkg2.cake_backend"),
    )
    for name in cake_backend._DET_ENTRIES:
        setattr(fake, name, Recorder())
    monkeypatch.setitem(sys.modules, "vendored_det_pkg2", fake_pkg)
    monkeypatch.setitem(sys.modules, "vendored_det_pkg2.cake_backend", fake)
    monkeypatch.setattr(cake_jit, "_det_package", cake_jit._det_package)
    cake_backend.use_deterministic_backend(fake)
    assert cake_backend.deterministic_backend() is fake
    replaced = ModuleType("vendored_det_pkg2.cake_backend")
    monkeypatch.setitem(sys.modules, "vendored_det_pkg2.cake_backend", replaced)
    assert cake_backend.deterministic_backend() is replaced
    cake_backend.use_deterministic_backend(det_backend)
    assert cake_backend.deterministic_backend() is det_backend


def test_use_deterministic_backend_drops_the_convenience_entries(monkeypatch):
    """A binding drops this package's convenience entries and arenas: an entry holds the plan and program of the
    package bound before, so a call with an unchanged key would launch that package's program."""
    fake_pkg, fake = (
        ModuleType("vendored_det_pkg3"),
        ModuleType("vendored_det_pkg3.cake_backend"),
    )
    for name in cake_backend._DET_ENTRIES:
        setattr(fake, name, Recorder())
    monkeypatch.setitem(sys.modules, "vendored_det_pkg3", fake_pkg)
    monkeypatch.setitem(sys.modules, "vendored_det_pkg3.cake_backend", fake)
    monkeypatch.setattr(cake_jit, "_det_package", cake_jit._det_package)
    caches = (
        cake_backend._K1_ATOMIC_CACHE,
        cake_backend._K1_DET_CACHE,
        cake_backend._K2_ATOMIC_CACHE,
        cake_backend._K2_SPLITK_CACHE,
    )
    key = ("dense-projection-gemm-test", object())
    try:
        for module in (fake, det_backend):
            for cache in caches:
                cache[key] = None
            cake_backend._ATOMIC_ARENAS[(99, 99)] = cake_backend._AtomicArena(
                torch.device("cpu"), (99, 99)
            )
            cake_backend.use_deterministic_backend(module)
            assert cake_backend.deterministic_backend() is module
            assert not any(key in cache for cache in caches)
            assert (99, 99) not in cake_backend._ATOMIC_ARENAS
    finally:
        for cache in caches:
            cache.pop(key, None)
        cake_backend._ATOMIC_ARENAS.pop((99, 99), None)
        cake_backend.use_deterministic_backend(det_backend)


@pytest.mark.parametrize("shift_elems, ldo", [(1, 256), (2, 256), (0, 258)])
def test_k2_misaligned_out_takes_the_deterministic_path(monkeypatch, shift_elems, ldo):
    """An ``out`` the reduce-add map cannot address (off a 16-byte boundary, or rows not a multiple of 4 elements)
    is not planned here: the call takes the deterministic path, which serves or refuses it as that package does."""
    arch, T, K, N = "sm_103a", 4099, 6144, 256
    _register(
        monkeypatch, arch, "router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"
    )
    X, W = torch.zeros(T, K), torch.zeros(N, K)
    storage = torch.zeros(T * ldo + 16)
    out = storage[shift_elems : shift_elems + T * ldo].view(T, ldo)[:, :N]
    plan, reason = cake_backend.plan_atomic_router(
        X, W.t(), out, arch=arch, sm_count=152
    )
    assert plan is None and "16-byte" in reason
    det = _patch_det(monkeypatch, "cake_dense_gemm_nondet_train_router_plan")
    monkeypatch.setattr(
        cake_backend,
        "_device_and_arch",
        lambda A, B, o: (torch.device("cpu"), arch, 152),
    )
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_plan(X, W.t(), out)
        is det.result
    )
    ((args, kwargs),) = det.calls
    assert args[0] is X and args[1].data_ptr() == W.data_ptr() and args[2] is out
    assert kwargs == {"splits": None}


# ---------------------------------------------------------------------------
# PyTorch's deterministic-algorithms mode (CPU): the alert before an order-varying launch
# ---------------------------------------------------------------------------


def _prepared_launch(monkeypatch, k: str, order_varying: bool):
    """A prepared launch of this package on CPU (nothing is loaded or launched) and the recorders of what it issues:
    K1 (``"k1"``) at three splits (the partials added in the order the splits finish) or two (one add), its program;
    K2 (``"k2"``) at 120 units over 40 tiles (three partials per element) or 80 (two), its split, the fill of ``out``
    and its program."""
    monkeypatch.setattr(
        det_backend,
        "_ffi_stream_context",
        lambda index, raw=None: contextlib.nullcontext(),
    )
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 0)
    out, program = torch.zeros(8, 8), Recorder()
    if k == "k1":
        plan = cake_backend.CakeDenseGemmNondetProjectionPlan(
            L=1,
            M=4096,
            N=128,
            K=1000,
            out_f32=True,
            transposed_out=True,
            block_n=128,
            splits=3 if order_varying else 2,
            m_tiles=16,
            n_tiles=1,
            sm_pairs=76,
            template="k1_program",
        )
        launch = cake_backend.CakeDenseGemmNondetProjectionLaunch(
            "k1_module", "sm_103a", plan, out, {}, _launch=program
        )
        return launch, (program,)
    plan = cake_backend.CakeDenseGemmNondetRouterPlan(
        M=5120,
        N=256,
        K=192,
        a_mn=True,
        b_mn=True,
        swapped=False,
        layout="nn_t",
        m_tiles=40,
        n_tiles=2,
        k_iters=6,
        units=120 if order_varying else 80,
        template="k2_program",
    )
    split, fill = Recorder(), Recorder()
    launch = cake_backend.CakeDenseGemmNondetRouterLaunch(
        "k2_module", "sm_103a", plan, out, {}, _launch=program, _split=split, _fill=fill
    )
    return launch, (split, fill, program)


#: The words of the alert that name the order-varying launch of :func:`_prepared_launch`.
PREPARED_ALERT_TEXT = {"k1": "at 3 splits", "k2": "with 120 units over 40 tiles"}


@pytest.mark.parametrize("k", ["k1", "k2"])
def test_an_order_varying_prepared_launch_alerts_under_torch_deterministic_mode(
    monkeypatch, torch_determinism, k
):
    """With ``torch.use_deterministic_algorithms(True)`` a prepared launch whose result varies run to run (K1 at
    three splits, K2 with three partials per element) raises PyTorch's ``RuntimeError`` before it issues anything (no
    split, no fill of ``out``, no program); with ``warn_only=True`` it warns, at the caller's line, and launches; with
    the mode off it launches without a warning.  The alert names the launch and the deterministic package."""
    launch, recorders = _prepared_launch(monkeypatch, k, True)
    assert not launch.plan.reproducible
    text = PREPARED_ALERT_TEXT[k]
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match=re.escape(text)) as info:
        launch.launch()
    assert str(info.value).startswith("flashinfer.cake_dense_gemm_nondet_train: ")
    assert "flashinfer.cake_dense_gemm_det_train is the deterministic module" in str(
        info.value
    )
    assert "warn_only=True" in str(info.value)
    assert not any(r.calls for r in recorders)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match=re.escape(text)) as record:
        assert launch.launch() is launch.out
    assert all(len(r.calls) == 1 for r in recorders)
    (alert,) = [w for w in record if text in str(w.message)]
    assert Path(alert.filename).name == Path(__file__).name
    torch.use_deterministic_algorithms(False)
    with _no_warnings():
        assert launch.launch() is launch.out
    assert all(len(r.calls) == 2 for r in recorders)


@pytest.mark.parametrize("k", ["k1", "k2"])
def test_a_fixed_order_prepared_launch_does_not_alert(
    monkeypatch, torch_determinism, k
):
    """A prepared launch whose result repeats (K1 at two splits, one add of two partials; K2 with at most two partials
    per element; a SIMT program, which stores every element once) launches without an alert under
    ``torch.use_deterministic_algorithms(True)``."""
    launch, recorders = _prepared_launch(monkeypatch, k, False)
    assert launch.plan.reproducible
    if k == "k2":
        assert dataclasses.replace(launch.plan, units=0, simt=(64, 64)).reproducible
    torch.use_deterministic_algorithms(True)
    with _no_warnings():
        assert launch.launch() is launch.out
    assert all(len(r.calls) == 1 for r in recorders)
    if k == "k1":
        # the cluster program at three splits: its splits are added in cluster-rank order
        launch.plan = dataclasses.replace(launch.plan, splits=3, dsm=True)
        assert launch.plan.reproducible
        with _no_warnings():
            assert launch.launch() is launch.out
        assert all(len(r.calls) == 2 for r in recorders)


def test_the_cluster_program_does_not_alert(monkeypatch, torch_determinism):
    """The shipped cluster row ``5120 -> 128`` FP32 at 16384 tokens on a 148-SM ``sm_100a`` part (three splits, added in
    cluster-rank order) runs without an alert under ``torch.use_deterministic_algorithms(True)``: the planning call,
    the call from the cached key and the prepared launch (CPU binding, nothing loaded or launched)."""
    table = cake_backend.K1_DSM_RANGES_BY_SM
    entry_fn = _cpu_k1_launches(monkeypatch, "sm_100a")
    facts = det_backend.DeviceFacts(
        index=0, arch="sm_100a", sm_count=148, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    # the harness takes the cluster ranges out; this test runs the shipped one
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", table)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16, T = torch.bfloat16, 16384
    X, G = torch.empty(T, 5120, dtype=bf16), torch.empty(T, 128, dtype=bf16)
    out_w = torch.empty(128, 5120)
    torch.use_deterministic_algorithms(True)
    with _no_warnings():
        for _ in range(2):
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out_w)
                is out_w
            )
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
            G, X, out_w
        )
        assert prepared.launch() is out_w
    ((_, atomic),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert atomic.plan.dsm and atomic.plan.splits == 3 and not atomic.order_varying
    assert prepared.plan.dsm and prepared.plan.reproducible
    assert len(entry_fn.calls) == 3


@pytest.mark.parametrize("splits", [2, 3])
def test_k1_convenience_launch_alerts_only_when_order_varying(
    monkeypatch, torch_determinism, splits
):
    """``cake_dense_gemm_nondet_train_projection_wgrad()`` on a row this package's planner takes at ``splits`` splits (CPU
    binding, nothing loaded or launched), under ``torch.use_deterministic_algorithms(True)``: at three splits the
    planning call and the call from the cached key both raise before the program; with ``warn_only=True`` the call
    warns and launches; at two splits both calls launch without an alert.  The cached launch keeps whether its plan
    varies run to run."""
    entry_fn = _cpu_k1_launches(monkeypatch, "sm_103a")
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    monkeypatch.setattr(cake_backend, "k1_plan", lambda **kw: (128, splits))
    monkeypatch.setattr(cake_backend, "k1_chain_plan", lambda **kw: None)
    bf16, T = torch.bfloat16, 1000
    G, X = torch.empty(T, 128, dtype=bf16), torch.empty(T, 4096, dtype=bf16)
    out = torch.zeros(128, 4096)

    def call():
        return cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)

    torch.use_deterministic_algorithms(True)
    for _ in range(2):  # the planning call, then the call from the cached key
        if splits == 3:
            with pytest.raises(RuntimeError, match="at 3 splits"):
                call()
        else:
            with _no_warnings():
                assert call() is out
    ((_, entry),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert entry.plan.splits == splits and entry.k1 is not None
    assert entry.order_varying == (splits == 3)
    assert len(entry_fn.calls) == (0 if splits == 3 else 2)
    if splits == 3:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="at 3 splits"):
            assert call() is out
        assert len(entry_fn.calls) == 1


@pytest.mark.parametrize("splits", [2, 3])
def test_k2_convenience_launch_alerts_only_when_order_varying(
    monkeypatch, torch_determinism, splits
):
    """``cake_dense_gemm_nondet_train_router_wgrad(splits=splits)`` (CPU binding, nothing loaded or launched) under
    ``torch.use_deterministic_algorithms(True)``: three units per tile (three partials per element) raise before the
    split, the fill of ``out`` and the program, on the planning call and from the cached key; with ``warn_only=True``
    the call warns and launches; two units per tile launch without an alert."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch)
    _sm103a_wgrad_stream_k(monkeypatch)
    split = Recorder()  # the torch chain's split
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    G, X, out = torch.empty(192, 256), torch.empty(192, 5120), torch.zeros(256, 5120)
    recorders = (entry_fn, fill, split)

    def call():
        return cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
            G, X, out, splits=splits
        )

    torch.use_deterministic_algorithms(True)
    for _ in range(2):  # the planning call, then the call from the cached key
        if splits == 3:
            with pytest.raises(RuntimeError, match="does not have a deterministic"):
                call()
        else:
            with _no_warnings():
                assert call() is out
    ((_, entry),) = [
        (k, e) for k, e in cake_backend._K2_ATOMIC_CACHE.items() if e.plan is not None
    ]
    assert entry.plan.simt is None and entry.k2 is not None
    assert entry.order_varying == (splits == 3) == (not entry.plan.reproducible)
    assert any(r.calls for r in recorders) == (splits == 2)
    if splits == 3:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match=re.escape(entry.plan.template)):
            assert call() is out
        assert entry_fn.calls


def test_routes_table():
    assert cake_backend.ROUTES == {
        "k1_kk": "det",
        "k1_kn": "det",
        "k1_nk": "det",
        "k1_nn": "atomic",
        "k2_kk": "atomic",
        "k2_kn": "atomic",
        "k2_nn_t": "atomic",
        "split": "det",
    }


# ---------------------------------------------------------------------------
# Bounded convenience state, the router's FP32 contract and the alert of fixed-order launches (CPU)
# ---------------------------------------------------------------------------

#: The token counts of one K1 handle (the weights of one layer) whose ``T`` changes from call to call: ragged counts,
#: tile edges and the GLM-5.2 per-rank token counts.
K1_CHANGING_T = (255, 257, 1000, 1001, 2048, 4095, 4097, 8191, 8193, 12345, 16172, 16231, 16384)  # fmt: skip


def _k1_convenience_caches() -> list:
    """The K1 convenience caches of this package (every ``_K1_*_CACHE`` of ``cake_backend``)."""
    return [
        cache
        for name, cache in vars(cake_backend).items()
        if name.startswith("_K1_")
        and name.endswith("_CACHE")
        and isinstance(cache, dict)
    ]


def test_k1_convenience_cache_bounded_under_changing_tokens(monkeypatch):
    """One K1 handle called with a changing token count: every forward, input gradient (BF16 and FP32 out) and
    weight gradient (BF16 and FP32 out; the FP32 row with a verdict, so this package's split-K program and its slots
    and tickets) is a new key, yet each K1 convenience cache holds at most the deterministic backend's
    ``CONVENIENCE_CACHE_SIZE`` entries after every call.  The calls share one arena on their stream, whose bytes stay
    within four times the largest requirement of each workspace kind (a live buffer below twice it, the buffers a
    growth retired below the live one), and a second pass over the same token counts grows nothing (CPU binding,
    nothing loaded or launched)."""
    entry_fn = _cpu_k1_launches(monkeypatch, "sm_103a")
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    monkeypatch.setattr(det_backend, "CONVENIENCE_CACHE_SIZE", 4)
    key = ("sm_103a", 1, 4096, 128, "f32")
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (128, 2))
    monkeypatch.delitem(cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS, key, raising=False)
    largest: dict[str, int] = {}
    require = cake_backend._AtomicArena.require

    def recording_require(arena, kind, numel):
        need = max(int(numel), cake_backend._ATOMIC_ARENA_MIN_ELEMS)
        largest[kind] = max(largest.get(kind, 0), need)
        return require(arena, kind, numel)

    monkeypatch.setattr(cake_backend._AtomicArena, "require", recording_require)
    bf16, f32 = torch.bfloat16, torch.float32
    W = torch.empty(128, 4096, dtype=bf16)
    caches = _k1_convenience_caches()
    assert len(caches) >= 2
    calls = 0

    def sweep():
        nonlocal calls
        for T in K1_CHANGING_T:
            X, G = torch.empty(T, 4096, dtype=bf16), torch.empty(T, 128, dtype=bf16)
            launches = (
                lambda: cake_backend.cake_dense_gemm_nondet_train_projection_forward(
                    X, W, torch.empty(T, 128, dtype=bf16)
                ),
                lambda: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(
                    G, W, torch.empty(T, 4096, dtype=bf16)
                ),
                lambda: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(
                    G, W, torch.empty(T, 4096, dtype=f32)
                ),
                lambda: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
                    G, X, torch.empty(128, 4096, dtype=bf16)
                ),
                lambda: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
                    G, X, torch.empty(128, 4096, dtype=f32)
                ),
            )
            for launch in launches:
                launch()
                calls += 1
                assert all(
                    len(c) <= det_backend.CONVENIENCE_CACHE_SIZE for c in caches
                ), (T, [len(c) for c in caches])

    def arena_bound() -> int:
        """Four times the largest requirement of every kind the arena served, in bytes."""
        kinds = cake_backend._ATOMIC_ARENA_KINDS
        return sum(
            4 * n * torch.empty(0, dtype=kinds[kind][0]).element_size()
            for kind, n in largest.items()
        )

    sweep()
    assert len(entry_fn.calls) == calls == 5 * len(K1_CHANGING_T)
    assert list(cake_backend._ATOMIC_ARENAS) == [(0, 0)]
    (arena,) = cake_backend._ATOMIC_ARENAS.values()
    # the weight gradient's split-K program and the deterministic programs' stream-K workspace both used the arena
    assert {"k1_slots", "k1_tickets"} <= set(largest), sorted(largest)
    assert 0 < arena.nbytes() <= arena_bound(), (arena.nbytes(), largest)
    nbytes, version = arena.nbytes(), arena.version
    sweep()
    assert len(entry_fn.calls) == calls
    assert list(cake_backend._ATOMIC_ARENAS) == [(0, 0)]
    assert cake_backend._ATOMIC_ARENAS[(0, 0)] is arena
    assert (arena.nbytes(), arena.version) == (nbytes, version)


# A router call on fresh CPU tensors: (op, the entry point's call of its three tensors, split count).  "gemm" is the
# GEMM entry point of the forward's views, "gemm_wgrad" of the weight gradient's (``G.T @ X``).
ROUTER_DTYPE_CASES = {
    "gemm": ("gemm", None),
    "gemm_two_splits": ("gemm", 2),
    "gemm_wgrad": ("gemm_wgrad", None),
    "plan": ("plan", None),
    "prepare": ("prepare", None),
    "forward": ("fwd", None),
    "forward_two_splits": ("fwd", 2),
    "dgrad": ("dgrad", None),
    "wgrad": ("wgrad", None),
    "wgrad_two_splits": ("wgrad", 2),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "second", "out"])
@pytest.mark.parametrize("case", sorted(ROUTER_DTYPE_CASES))
def test_router_refuses_tensors_that_are_not_fp32(monkeypatch, case, position, dtype):
    """The router GEMM is FP32 in and out: a first operand, second operand or ``out`` of another dtype raises
    ``ValueError`` at every router entry point -- the GEMM (both layout classes, with and without a split count), its
    plan and prepared launch, the forward (with and without a split count), the input gradient and the weight gradient
    (with and without one) -- before anything is launched: no BF16x3 split, no fill of ``out``, no program; nothing
    converts the tensor or runs a BF16 or TF32 GEMM instead (CPU binding, nothing loaded or launched)."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch)
    split = Recorder()
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    op, splits = ROUTER_DTYPE_CASES[case]
    T, H, E = 192, 5120, 256
    shapes = {
        "gemm": ((T, H), (E, H), (T, E)),
        "plan": ((T, H), (E, H), (T, E)),
        "prepare": ((T, H), (E, H), (T, E)),
        "fwd": ((T, H), (E, H), (T, E)),
        "gemm_wgrad": ((T, E), (T, H), (E, H)),
        "dgrad": ((T, E), (E, H), (T, H)),
        "wgrad": ((T, E), (T, H), (E, H)),
    }[op]
    P, Q, out = (
        torch.empty(shape, dtype=dtype if i == position else torch.float32)
        for i, shape in enumerate(shapes)
    )
    kw = {} if splits is None else {"splits": splits}
    call = {
        "gemm": lambda: cake_backend.cake_dense_gemm_nondet_train_router(P, Q.t(), out, **kw),
        "plan": lambda: cake_backend.cake_dense_gemm_nondet_train_router_plan(P, Q.t(), out),
        "prepare": lambda: cake_backend.cake_dense_gemm_nondet_train_router_prepare(P, Q.t(), out),
        "gemm_wgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router(P.t(), Q, out, **kw),
        "fwd": lambda: cake_backend.cake_dense_gemm_nondet_train_router_forward(P, Q, out, **kw),
        "dgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router_dgrad(P, Q, out),
        "wgrad": lambda: cake_backend.cake_dense_gemm_nondet_train_router_wgrad(P, Q, out, **kw),
    }[op]  # fmt: skip
    for _ in range(2):  # the first call and a call with the same tensors again
        with pytest.raises(ValueError, match="must be fp32"):
            call()
    assert not entry_fn.calls and not fill.calls and not split.calls
    assert out.dtype == (dtype if position == 2 else torch.float32)


def test_router_planner_refuses_tensors_that_are_not_fp32():
    """:func:`plan_atomic_router` takes FP32 views only: a BF16, FP16 or FP64 operand or ``out`` raises
    ``ValueError`` on every layout class, before the planner reads a rule table."""
    T, H, E = 64, 5120, 256
    views = {
        "kk": lambda d: (torch.empty(T, H, dtype=d[0]), torch.empty(E, H, dtype=d[1]).t(), torch.empty(T, E, dtype=d[2])),
        "kn": lambda d: (torch.empty(T, E, dtype=d[0]), torch.empty(E, H, dtype=d[1]), torch.empty(T, H, dtype=d[2])),
        "nn_t": lambda d: (torch.empty(T, E, dtype=d[0]).t(), torch.empty(T, H, dtype=d[1]), torch.empty(E, H, dtype=d[2])),
    }  # fmt: skip
    f32 = torch.float32
    for layout, make in views.items():
        plan, _ = cake_backend.plan_atomic_router(
            *make((f32, f32, f32)), arch="sm_103a", sm_count=152, splits=2
        )
        assert plan is None or plan.layout == layout, layout
        for other in (torch.bfloat16, torch.float16, torch.float64):
            for position in range(3):
                dtypes = [f32, f32, f32]
                dtypes[position] = other
                with pytest.raises(ValueError, match="must be fp32"):
                    cake_backend.plan_atomic_router(
                        *make(dtypes), arch="sm_103a", sm_count=152, splits=2
                    )


def _single_k1_entry():
    """The one entry of the K1 convenience caches (each test call starts from empty caches)."""
    entries = [e for cache in _k1_convenience_caches() for e in cache.values()]
    assert len(entries) == 1, len(entries)
    return entries[0]


def _call_under_the_mode(launch) -> RuntimeError | None:
    """``launch()`` under ``torch.use_deterministic_algorithms(True)``: the ``RuntimeError`` of PyTorch's alert, or
    ``None`` after a launch that raised and warned nothing (the mode is off again afterwards)."""
    torch.use_deterministic_algorithms(True)
    try:
        with _no_warnings():
            launch()
    except RuntimeError as exc:
        assert "does not have a deterministic implementation" in str(exc), exc
        return exc
    finally:
        torch.use_deterministic_algorithms(False)
    return None


def test_k1_forward_and_input_gradient_alert_only_when_order_varying(
    monkeypatch, torch_determinism
):
    """Under ``torch.use_deterministic_algorithms(True)`` the forward and input-gradient calls -- the entry points of
    the forward and of the input gradient (BF16 and FP32 out), the GEMM entry point of their layouts and the prepared
    launches -- alert exactly before a launch whose result can vary run to run: a call whose cached launch repeats bit
    for bit (``order_varying`` false), and a prepared launch of a deterministic program or of a reproducible plan, runs
    without an alert or a warning on its planning call and from its cached key, and launches (CPU binding, nothing
    loaded or launched)."""
    entry_fn = _cpu_k1_launches(monkeypatch, "sm_103a")
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16, f32 = torch.bfloat16, torch.float32
    fixed = varying = 0
    for K, N in ((4096, 128), (4096, 512), (6144, 2048)):
        for T in (1000, 4097, 16231):
            X, W, G = (
                torch.empty(T, K, dtype=bf16),
                torch.empty(N, K, dtype=bf16),
                torch.empty(T, N, dtype=bf16),
            )
            out_f = torch.empty(T, N, dtype=bf16)
            out_d, out_d32 = torch.empty(T, K, dtype=bf16), torch.empty(T, K, dtype=f32)
            # the entry point's call and the prepared launch of the same GEMM
            cases = (
                (lambda: cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, out_f), (X, W.t(), out_f)),
                (lambda: cake_backend.cake_dense_gemm_nondet_train_projection(X, W.t(), out_f), (X, W.t(), out_f)),
                (lambda: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out_d), (G, W, out_d)),
                (lambda: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out_d32), (G, W, out_d32)),
                (lambda: cake_backend.cake_dense_gemm_nondet_train_projection(G, W, out_d32), (G, W, out_d32)),
            )  # fmt: skip
            # a row of K1_FORWARD_PROGRAM_RULES_BY_SM / K1_DGRAD_PROGRAM_RULES_BY_SM on this part runs this package's
            # program, bound once per key in _K1_PROGRAM_CACHE (the generic entry point reaches the entry point
            # through _k1_table_call); the same bits on every run, so never an alert
            forward = cake_backend._k1_forward_program_of(X, W, out_f)
            dgrad32 = cake_backend._k1_dgrad_program_of(G, W, out_d32)
            programs = (
                forward,
                forward,
                cake_backend._k1_dgrad_program_of(G, W, out_d),
                dgrad32,
                dgrad32,
            )
            for (call, views), program in zip(cases, programs, strict=True):
                cake_backend._drop_convenience_entries()
                # the planning call, then the call from the cached key
                for _ in range(2):
                    before = len(entry_fn.calls)
                    alert = _call_under_the_mode(call)
                    if program is not None:
                        assert alert is None, (K, N, T, program)
                        own = cake_backend._K1_PROGRAM_CACHE
                        caches = [c for c in _k1_convenience_caches() if c is not own]
                        assert not any(list(c.values()) for c in caches), (K, N, T)
                        names = [e.module_name for e in own.values()]
                        assert names == [program], (K, N, T)
                        assert not any(e.order_varying for e in own.values())
                        assert len(entry_fn.calls) == before + 1
                        continue
                    entry = _single_k1_entry()
                    assert (alert is not None) == bool(entry.order_varying), (K, N, T)
                    assert len(entry_fn.calls) == before + (alert is None)
                order_varying = program is None and bool(entry.order_varying)
                fixed += not order_varying
                varying += order_varying
                prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
                    *views
                )
                plan = getattr(prepared, "plan", None)
                order_varying = (
                    isinstance(
                        prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
                    )
                    and not plan.reproducible
                )
                before = len(entry_fn.calls)
                alert = _call_under_the_mode(prepared.launch)
                assert (alert is not None) == order_varying, (K, N, T)
                assert len(entry_fn.calls) == before + (alert is None)
    print(
        f"K1_FWD_DGRAD_ALERT fixed-order calls {fixed}, order-varying calls {varying}"
    )


# (op, arch, T, hidden): a router call in a SIMT window of the shipped router tables on a 152-SM part.
ROUTER_SIMT_CALLS = {
    "fwd": ("sm_103a", 64, 5120),
    "dgrad": ("sm_100a", 64, 5120),
    "wgrad": ("sm_103a", 64, 5120),
}


@pytest.mark.parametrize("op", sorted(ROUTER_SIMT_CALLS))
def test_router_simt_launches_do_not_alert(monkeypatch, torch_determinism, op):
    """The router's SIMT programs store every element of ``out`` once, their partial sums merged in a fixed order, so
    their output repeats bit for bit: under ``torch.use_deterministic_algorithms(True)`` the entry point's planning
    call, its calls from the cached key and the prepared launch of the same GEMM run the SIMT program without an alert
    or a warning (no split, no fill of ``out``; CPU binding, nothing loaded or launched)."""
    arch, T, H = ROUTER_SIMT_CALLS[op]
    entry_fn, fill = _cpu_k2_launches(monkeypatch, arch)
    split = Recorder()
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    E = 256
    X, W, G = torch.empty(T, H), torch.empty(E, H), torch.empty(T, E)
    P, Q, out, A, B = {
        "fwd": (X, W, torch.empty(T, E), X, W.t()),
        "dgrad": (G, W, torch.empty(T, H), G, W),
        "wgrad": (G, X, torch.empty(E, H), G.t(), X),
    }[op]
    call = {
        "fwd": cake_backend.cake_dense_gemm_nondet_train_router_forward,
        "dgrad": cake_backend.cake_dense_gemm_nondet_train_router_dgrad,
        "wgrad": cake_backend.cake_dense_gemm_nondet_train_router_wgrad,
    }[op]
    torch.use_deterministic_algorithms(True)
    with _no_warnings():
        for _ in range(3):  # the planning call, then the calls from the cached key
            assert call(P, Q, out) is out
        prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(A, B, out)
        assert prepared.launch() is out
    launches = [e for e in cake_backend._K2_ATOMIC_CACHE.values() if e.plan is not None]
    assert launches, "the call did not plan this package's program"
    for launch in launches:
        assert launch.plan.simt is not None and launch.plan.reproducible
        assert not launch.order_varying
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
    assert prepared.plan.simt is not None and prepared.plan.reproducible
    assert len(entry_fn.calls) == 4
    assert not fill.calls and not split.calls


# ---------------------------------------------------------------------------
# The benchmark's row classes, acceptance bar and reports (CPU)
# ---------------------------------------------------------------------------

BENCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "bench_cake_dense_gemm_nondet_train.py"
)


@functools.cache
def _bench() -> ModuleType:
    """The benchmark script as a module (its functions only: nothing runs on import).  A vendored run
    (``scripts/check_cake_build_manifests.py --vendored-gate``, which blocks flashinfer with
    ``sys.modules["flashinfer"] = None``) skips: the script imports flashinfer."""
    import importlib.util

    if sys.modules.get("flashinfer", False) is None:
        pytest.skip(
            "the benchmark script imports flashinfer, which the vendored gate blocks"
        )
    spec = importlib.util.spec_from_file_location(
        "bench_cake_dense_gemm_nondet_train", BENCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_row_classes():
    """Every row of the benchmark carries one class: ``router``, ``batched`` (the MLA head batch), ``large`` (a
    projection whose in and out features both exceed 1024) or ``small-output`` (every other projection: at most 1024
    out features, for its forward, input gradient and weight gradient alike)."""
    bench = _bench()
    rows = bench.row_table()
    seen = set()
    for label, spec in rows.items():
        klass = bench.row_class(spec)
        seen.add(klass)
        if spec["family"] == "router":
            assert klass == "router", label
        elif spec["family"] == "mla":
            assert klass == "batched", label
        elif min(spec["K"], spec["N"]) > 1024:
            assert klass == "large", label
        else:
            assert klass == "small-output", label
            assert spec["N"] <= 1024, label
    assert seen == set(bench.CLASSES)
    expected = {
        "4096to1024_fwd_bf16_t16231": "small-output",
        "kv_a_dgrad_bf16_t16231": "small-output",
        "indexer_hw_wgrad_f32_t16172": "small-output",
        "4096to4096_fwd_bf16_t16231": "large",
        "q_a_wgrad_f32_t16172": "large",
        "o_proj_dgrad_f32_t16231": "large",
        "mla_qabs_fwd_bf16_t16172": "batched",
        "mla_vproj_wgrad_bf16_t16231": "batched",
        "router_h5120_wgrad_t16231": "router",
    }
    assert {label: bench.row_class(rows[label]) for label in expected} == expected


def test_bench_acceptance_bar_is_three_percent():
    """A row ships this package's program only when every paired group is faster than the deterministic package
    and the median ratio is below 0.97 (at least 3 % faster)."""
    bench = _bench()
    assert bench.ACCEPT_RATIO == 0.97

    def accept(ratios):
        medians = {"eager": {"det": [1.0] * len(ratios), "atomic": list(ratios)}}
        verdict = bench.summarize(medians, medians, {})["eager"]["atomic_over_det"]
        return verdict["accept"]

    assert accept([0.96] * 5)
    assert not accept([0.975] * 5)
    assert not accept([0.99] * 5)
    assert not accept([0.9, 0.9, 0.9, 0.9, 1.01])  # one group slower


def _bench_entry(bench, label, ms):
    rows = bench.row_table()
    rec = dict(
        ms=ms,
        p5_ms=dict(ms),
        p95_ms=dict(ms),
        groups_ms={name: [value] for name, value in ms.items()},
        errors={},
        sota_arm="cublas",
        sota_ms=ms["cublas"],
        atomic_over_det=dict(median=ms["atomic"] / ms["det"], accept=False),
        atomic_shim_over_det=dict(median=ms["atomic_shim"] / ms["det"]),
    )
    return {
        "row": label,
        "spec": rows[label],
        "class": bench.row_class(rows[label]),
        "atomic_program": True,
        "accept": False,
        "atomic_plan": dict(template="program", reproducible=True),
        "bits_repeat": {
            name: dict(runs=5, equal=name != "cublas")
            for name in ("det", "atomic", "atomic_shim", "cublas")
        },
        "cupti": dict(timed_calls=10, fallbacks=0, note=None),
        "modes": {"graph": rec},
    }


def test_bench_reports_carry_the_class_shape_and_baselines():
    """A CSV line names the row's family, class, operation, K, N, batch, T, output dtype, architecture and SM count,
    with the plain cuBLAS and package-path latencies and the bit-repeat results; the per-class summary groups rows by
    class, operation and mode; the markdown report has both and says whether CUPTI timed every call."""
    bench = _bench()
    for field in (
        "family", "class", "op", "K", "N", "T", "dtype", "arch", "sm_count", "cublas_ms", "atomic_shim_ms",
        "atomic_bits_repeat", "cupti_fallbacks",
    ):  # fmt: skip
        assert field in bench.CSV_FIELDS
    ms = dict(det=2.0, atomic=1.0, atomic_shim=1.1, cublas=1.5)
    entry = _bench_entry(bench, "mla_vproj_dgrad_bf16_t16231", ms)
    header = dict(arch="sm_100a", sm_count=152)
    (row,) = bench.csv_rows(entry, ("graph",), header)
    assert set(row) <= set(bench.CSV_FIELDS)
    keys = ("family", "class", "op", "K", "N", "batch", "T", "dtype", "arch", "sm_count")  # fmt: skip
    want = ("mla", "batched", "dgrad", 512, 256, 64, 16231, "bf16", "sm_100a", 152)
    assert tuple(row[k] for k in keys) == want
    assert row["det_over_atomic"] == 2.0 and row["cublas_over_atomic"] == 1.5
    assert row["atomic_shim_over_atomic"] == pytest.approx(1.1)
    assert row["atomic_bits_repeat"] is True and row["cublas_bits_repeat"] is False
    assert row["bit_runs"] == 5 and row["cupti_fallbacks"] == 0
    other = _bench_entry(bench, "4096to128_fwd_bf16_t16231", dict(ms, det=4.0))
    summary = bench.class_summary([entry, other], ("graph",))
    assert [(g["class"], g["op"], g["mode"]) for g in summary] == [
        ("small-output", "fwd", "graph"),
        ("batched", "dgrad", "graph"),
    ]
    small = summary[0]
    assert (small["rows"], small["atomic_program"], small["accepted"]) == (1, 1, 0)
    assert (small["bits_checked"], small["bits_repeat"]) == (1, 1)
    assert small["det_over_atomic_geomean"] == pytest.approx(4.0)
    assert small["shim_over_atomic_geomean"] == pytest.approx(1.1)
    results = dict(
        device="GPU", arch="sm_100a", sm_count=152, versions={}, image={}, groups=1, steps=1, lt_algos=8,
        bit_runs=5, cupti=dict(timed_calls=20, fallbacks=0, note=None), rows=[entry, other],
    )  # fmt: skip
    text = bench.markdown(results, ("graph",))
    assert "cuBLAS (torch.matmul) latency" in text and "public entry point" in text
    assert "| small-output | fwd | graph | 1 | 1 | 0 | 1 of 1 |" in text
    assert "CUPTI timed all 20 timing calls." in text
    results["cupti"] = dict(timed_calls=20, fallbacks=2, note="no CUPTI")
    assert "2 of 20 timing calls fell back to CUDA events" in bench.markdown(
        results, ("graph",)
    )


# ---------------------------------------------------------------------------
# Registry, names and workspace (CPU)
# ---------------------------------------------------------------------------


def test_registry_is_disjoint_from_the_deterministic_package():
    """This package's programs and templates never equal a deterministic one, nor a template the deterministic
    package registered before its rename (its ``*_det_*`` templates with ``det`` removed)."""
    assert sorted(cake_jit.KERNELS) == sorted(cake_jit.ARCHES)
    det_hex = {name.rsplit("_", 1)[1] for name in det_jit.MODULES}
    det_templates = {t for table in det_jit.KERNELS.values() for t in table}
    former_templates = {t.replace("_det_", "_") for t in det_templates} | {
        "split_fp32_bf16x3"
    }
    for name, record in cake_jit.MODULES.items():
        assert re.fullmatch(r"cake_dense_projection_gemm_[0-9a-f]{20}", name), name
        assert record["library"] == name and record["kernel"] == f"kernel_{name}"
        assert name.rsplit("_", 1)[1] not in det_hex, name
        assert record["template"].startswith(cake_jit.TEMPLATE_PREFIXES), name
        for arch in record["arches"]:
            assert cake_jit.KERNELS[arch][record["template"]] == name
    templates = {t for table in cake_jit.KERNELS.values() for t in table}
    assert not templates & (det_templates | former_templates)
    for arch, table in cake_jit.KERNELS.items():
        for template, name in table.items():
            assert (
                arch in cake_jit.MODULES[name]["arches"]
                and cake_jit.MODULES[name]["template"] == template
            )


def test_registered_templates_are_the_ones_the_planners_name():
    """Every template this checkout registers is one the host plans: the K1 programs of each tile width
    (``dense_proj_gemm_atomic_nn_n<64|128|256>[_f32]_t`` and the row-major ``dense_proj_gemm_atomic_nn_n192[_f32]``)
    and their FP32 programs of several accumulation chains (``..._c<chains>``, :data:`K1_CHAINS_CHOICES`), the K1
    cluster programs the ranges of :data:`K1_DSM_RANGES_BY_SM` name (each registered for the architecture of its
    table, and for no other: ``dense_proj_gemm_dsm_nn_n<block_n>_f32_t_s<splits>``; also the cluster tiles of
    :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`; the three-CTA programs stay registered where no range names them: the ranges
    that ran them run two CTAs), the two router classes, the weight gradient's head/tail programs (of the architecture
    and of its SM counts) and K-order programs (:data:`ROUTER_KSUM_TEMPLATES_BY_SM`), the
    SIMT programs the windows of the weight gradient, the input gradient and the forward name (in :data:`ROUTER_RULES` and the
    architecture's tables of :data:`ROUTER_RULES_BY_SM`) and the partial sum of the deterministic router program's
    split-K partials; each SIMT program a window names is registered."""
    wanted = {
        cake_backend.atomic_k1_template(bn, f32)
        for bn in cake_backend.K1_BLOCK_N_CHOICES
        for f32 in (False, True)
    } | {
        "router_fp32_gemm_atomic_kk",
        "router_fp32_gemm_atomic_nn_t",
        "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX,
        cake_backend.PARTIALS_SUM_TEMPLATE,
    }
    wanted |= {
        cake_backend.atomic_k1_template(bn, True, chains)
        for bn in cake_backend.K1_BLOCK_N_CHOICES
        for chains in cake_backend.K1_CHAINS_CHOICES
    }
    # the programs of K1_ORDERED_RANGES_BY_SM and K1_PARITY_RANGES_BY_SM and the programs the forward /
    # input-gradient (this package's and the nvjet program on the deterministic plan) / stream-K weight-gradient
    # tables name, each registered for the architecture of its table
    named = {}
    for (arch, _), rows in cake_backend.K1_ORDERED_RANGES_BY_SM.items():
        for key, ranges in rows.items():
            for _, _, (bn, _, chains, order) in ranges:
                t = cake_backend.atomic_k1_template(bn, key[4] == "f32", chains, order)
                named.setdefault(arch, set()).add(t)
    for (arch, _), rows in cake_backend.K1_PARITY_RANGES_BY_SM.items():
        for key, ranges in rows.items():
            for _, _, *programs in ranges:
                for program in programs:
                    if program:
                        bn, _, chains, order = program
                        t = cake_backend.atomic_k1_template(
                            bn, key[4] == "f32", chains, order
                        )
                        named.setdefault(arch, set()).add(t)
    for table in (
        cake_backend.K1_FORWARD_PROGRAM_RULES_BY_SM,
        cake_backend.K1_DGRAD_PROGRAM_RULES_BY_SM,
        cake_backend.K1_DGRAD_NVJET_RULES_BY_SM,
        cake_backend.K1_WGRAD_SK_RULES_BY_SM,
    ):
        for (arch, _), rows in table.items():
            named.setdefault(arch, set()).update(rows.values())
    assert set(named) == {"sm_100a", "sm_103a", "sm_107a"}
    for arch, templates in named.items():
        assert templates <= set(cake_jit.KERNELS[arch]), (
            arch,
            sorted(templates - set(cake_jit.KERNELS[arch])),
        )
        wanted |= templates
    dsm = {
        arch: {
            cake_backend.dsm_k1_template(bn, s)
            for (table_arch, _), rows in cake_backend.K1_DSM_RANGES_BY_SM.items()
            if table_arch == arch
            for ranges in rows.values()
            for _, _, (bn, s) in ranges
        }
        | {
            cake_backend.dsm_k1_template(*program)
            for (table_arch, _), rows in (
                cake_backend.K1_DSM_ODD_BLOCK_RANGES_BY_SM.items()
            )
            if table_arch == arch
            for ranges in rows.values()
            for _, _, program in ranges
            if program is not None and len(program) == 2
        }
        for arch in cake_jit.KERNELS
    }
    assert dsm["sm_100a"] and dsm["sm_103a"]
    three = {cake_backend.dsm_k1_template(bn, 3) for bn in (32, 64, 128)}
    for arch, table in cake_jit.KERNELS.items():
        assert dsm[arch] <= set(table), (arch, sorted(dsm[arch] - set(table)))
        registered = {t for t in table if t.startswith(cake_jit.TEMPLATE_PREFIXES[7])}
        assert registered - dsm[arch] <= three, (arch, sorted(registered - dsm[arch]))
    by_sm = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES_BY_SM
    rule_tables = (
        cake_backend.ROUTER_RULES,
        *cake_backend.ROUTER_RULES_BY_SM.values(),
    )
    for arch, table in cake_jit.KERNELS.items():
        ksum = {
            template
            for (
                sm_arch,
                _,
            ), template in cake_backend.ROUTER_KSUM_TEMPLATES_BY_SM.items()
            if sm_arch == arch
        }
        assert ksum <= set(table), (arch, sorted(ksum - set(table)))
        heads = (
            {cake_backend.router_head_tail_template(arch)}
            | {template for (sm_arch, _), template in by_sm.items() if sm_arch == arch}
            | ksum
        )
        simt = {
            cake_backend.router_simt_template(
                cake_backend.router_simt_tile(spec), key[1]
            )
            for rules in rule_tables
            for key, rule in rules.items()
            if key[0] == arch and isinstance(rule, tuple)
            for _, _, spec in rule
            if cake_backend.router_simt_tile(spec) is not None
        } | {
            cake_backend.router_forward_simt_template(
                cake_backend.router_forward_simt_tile(spec)
            )
            for rules in rule_tables
            for key, rule in rules.items()
            if key[0] == arch and isinstance(rule, tuple)
            for _, _, spec in rule
            if cake_backend.router_forward_simt_tile(spec) is not None
        }
        assert simt <= set(table), (arch, sorted(simt - set(table)))
        assert set(table) <= wanted | heads | simt | dsm[arch] | three, sorted(
            set(table) - wanted
        )


def test_no_det_token_in_generated_names():
    """No generated name of this package -- file, program, kernel, template, library -- carries the token
    ``det``."""
    root = Path(cake_jit.__file__).parent
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and "aot" not in p.parts
    ]
    for path in files:
        rel = path.relative_to(root).as_posix()
        assert not DET_TOKEN.search(path.name), rel
    generated = [*cake_jit.MODULES, *(r["kernel"] for r in cake_jit.MODULES.values())]
    generated += [t for table in cake_jit.KERNELS.values() for t in table]
    assert not [n for n in generated if DET_TOKEN.search(n)]
    manifest = root / "build.json"
    if manifest.exists():
        document = json.loads(manifest.read_text())
        # the in-tree import path: a vendored copy keeps its manifest
        assert document["package"] == (
            "flashinfer.experimental.cake.cake_dense_gemm_nondet_train"
        )
        for entry in document["libraries"]:
            assert not DET_TOKEN.search(entry["name"])
            assert all(
                s.startswith("csrc/cake_dense_projection_gemm/")
                or s == "csrc/cake_ffi_checks.h"
                or s.startswith("csrc/cake_device_helpers/")
                for s in entry["sources"]
            )


def test_select_module_and_the_loader_take_this_package_s_programs_only(monkeypatch):
    """``select_module``, the loader and ``bind_launch`` of this package serve its own programs only: a template or a
    program of the deterministic package is not forwarded to that package (``select_module`` raises, the loader goes
    to this package's own loading, ``bind_launch`` finds no record)."""
    for arch, table in det_jit.KERNELS.items():
        template = sorted(table)[0]
        with pytest.raises(NotImplementedError, match="not registered"):
            cake_jit.select_module(arch, template)
    with pytest.raises(NotImplementedError, match="not registered"):
        cake_jit.select_module("sm_100a", "dense_proj_gemm_atomic_nn_unregistered")
    det_loader, own_loader, det_bind = Recorder(), Recorder(), Recorder()
    monkeypatch.setattr(
        det_jit, "load_cake_dense_projection_det_gemm_module", det_loader
    )
    monkeypatch.setattr(cake_jit, "_load_module", own_loader)
    monkeypatch.setattr(det_backend, "bind_launch", det_bind)
    name = sorted(det_jit.MODULES)[0]
    assert name not in cake_jit.MODULES
    assert (
        cake_jit.load_cake_dense_projection_gemm_module(name, "sm_100a")
        is own_loader.result
    )
    assert own_loader.calls == [((name, "sm_100a"), {})]
    with pytest.raises(KeyError):
        cake_backend.bind_launch(name, {"M": 1}, (2, 1, 1), "sm_100a")
    assert not det_loader.calls and not det_bind.calls


def test_clear_convenience_caches_clears_both_packages():
    key = ("dense-projection-gemm-test", object())
    det_backend._K2_CACHE[key] = None
    cake_backend._K1_ATOMIC_CACHE[key] = None
    cake_backend._K1_DET_CACHE[key] = None
    cake_backend._K2_SPLITK_CACHE[key] = None
    cake_backend._ATOMIC_ARENAS[(99, 99)] = cake_backend._AtomicArena(
        torch.device("cpu"), (99, 99)
    )
    try:
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        assert (
            key not in det_backend._K2_CACHE
            and key not in cake_backend._K1_ATOMIC_CACHE
            and key not in cake_backend._K1_DET_CACHE
            and key not in cake_backend._K2_SPLITK_CACHE
        )
        assert (99, 99) not in cake_backend._ATOMIC_ARENAS
    finally:
        det_backend._K2_CACHE.pop(key, None)
        cake_backend._K1_ATOMIC_CACHE.pop(key, None)
        cake_backend._K1_DET_CACHE.pop(key, None)
        cake_backend._K2_SPLITK_CACHE.pop(key, None)
        cake_backend._ATOMIC_ARENAS.pop((99, 99), None)


def test_the_split_k_workspace_kinds():
    """The K1 tickets and the K1 slots are zeroed when allocated or grown (the last split of a partial tile reads
    slot columns no split writes); the kinds are not the deterministic package's."""
    kinds = cake_backend._ATOMIC_ARENA_KINDS
    assert kinds["k1_tickets"] == (torch.uint32, True)
    assert kinds["k1_slots"] == (torch.float32, True)
    # the deterministic K2 program's partials: written by the program before the reduction reads them
    assert kinds["k2_splitk_partials"] == (torch.float32, False)
    # the deterministic K1 program's stream-K slabs (no state between launches) and its slice counters (the
    # deterministic package's kinds: zero when allocated, zero again after every complete launch)
    assert kinds["k1_det_ws"] == det_backend._ARENA_KINDS["k1_ws"]
    assert kinds["k1_det_counters"] == det_backend._ARENA_KINDS["k1_counters"]
    assert not set(kinds) & set(det_backend._ARENA_KINDS)
    arena = cake_backend._AtomicArena(torch.device("cpu"), (0, 0))
    tickets = arena.require("k1_tickets", 100)
    assert tickets.dtype == torch.uint32 and tickets.numel() == 100
    assert not tickets.to(torch.int64).any()
    assert arena.require("k1_tickets", 50) is tickets and arena.version == 1
    grown = arena.require("k1_tickets", 1000)
    assert grown.numel() == 1000 and not grown.to(torch.int64).any()
    assert arena.version == 2
    slots = arena.require("k1_slots", 3)
    assert slots.dtype == torch.float32 and slots.numel() == 16 and not slots.any()
    grown = arena.require("k1_slots", 300)
    assert grown.numel() == 300 and not grown.any()


def test_arena_growth_retires_the_replaced_buffers():
    """A growth at least doubles a kind's buffer and retires the buffer it replaces (a CUDA graph or a launch in
    flight may still use it): the retired buffers keep their storage and contents until ``cake_dense_gemm_nondet_train_convenience_cache_clear``
    drops the arena, and hold fewer bytes than the live one; a requirement the live buffer meets grows nothing."""
    arena = cake_backend._AtomicArena(torch.device("cpu"), (0, 0))
    first = arena.require("k1_tickets", 100)
    first.view(torch.int32).fill_(7)  # stands for the state of a launch bound to it
    assert not arena.retired and arena.version == 1
    grown = arena.require("k1_tickets", 101)  # just larger: doubled
    assert grown.numel() == 200 and not grown.to(torch.int64).any()
    assert [id(b) for b in arena.retired] == [id(first)]
    assert bool((first.view(torch.int32) == 7).all())
    assert arena.require("k1_tickets", 150) is grown and arena.version == 2
    larger = arena.require("k1_tickets", 1000)  # more than doubled: the requirement
    assert larger.numel() == 1000
    assert [id(b) for b in arena.retired] == [id(first), id(grown)]
    for n in range(1001, 100_000, 997):  # a ragged requirement that keeps rising
        arena.require("k1_tickets", n)
    sizes = [int(b.numel()) for b in arena.retired]
    assert sizes == [100, 200, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    assert sum(sizes) < arena.buffers["k1_tickets"].numel() == 128000
    stack = arena.require("k2_stack", 3 * 64 * 256)
    assert arena.require("k2_stack", 3 * 65 * 256).numel() == 2 * stack.numel()
    assert arena.retired[-1] is stack
    assert arena.nbytes() == sum(
        int(b.numel()) * b.element_size()
        for b in (*arena.buffers.values(), *arena.retired)
    )


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _det_device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or not det_backend.generated_program_available(
        torch.device("cuda")
    ):
        pytest.skip(
            "requires a device the deterministic package registers programs for"
        )
    return torch.device("cuda")


def _atomic_device_or_skip() -> tuple[torch.device, str]:
    """A device this checkout registers programs of this package for.  A checkout without any (before the generated
    sources are added) skips; one that registers some but not a template a test names fails in
    :func:`_require_template`."""
    device = _det_device_or_skip()
    arch = det_backend.arch_for(device)
    if not cake_jit.KERNELS.get(arch):
        pytest.skip(f"this checkout registers no program of this package for {arch}")
    return device, arch


def _require_template(arch: str, template: str) -> None:
    assert template in cake_jit.KERNELS[arch], (
        f"{template} is not registered for {arch}"
    )


def _bf16(shape, device, generator, scale=1.0):
    return (torch.randn(shape, device=device, generator=generator) * scale).to(
        torch.bfloat16
    )


def _rel_fro(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref).norm() / ref.norm())


def _profile_kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]


# The router's BF16x3 split stage is the deterministic package's program by design (both modes run it); every
# other program of that package is a K1 / K2 GEMM.
_DET_SHARED_KERNELS = frozenset(
    module["kernel"]
    for module in det_jit.MODULES.values()
    if module["template"].startswith("split_fp32_det_bf16x3")
)


# The generated stem of the deterministic package's programs: its library names ``cake_<stem>_<hex>``, kernel
# symbols ``kernel_cake_<stem>_<hex>`` and template bodies' namespace ``cake::<stem>::``.
(_DET_STEM,) = {
    name.rsplit("_", 1)[0].removeprefix("cake_") for name in det_jit.MODULES
} or {"no deterministic program registered"}


def _det_kernel(name: str) -> bool:
    """A K1 / K2 GEMM kernel of the deterministic package (its programs' symbols and its template bodies'
    namespace); the shared BF16x3 split stage is not one."""
    return _DET_STEM in name and not any(
        kernel in name for kernel in _DET_SHARED_KERNELS
    )


def _assert_ran(fn, module_name: str) -> None:
    """``fn`` launches this package's program ``module_name`` and no K1 / K2 program of the deterministic
    package."""
    names = _profile_kernel_names(fn)
    kernel = cake_jit.MODULES[module_name]["kernel"]
    assert any(kernel in name for name in names), (kernel, names)
    assert not any(_det_kernel(name) for name in names), names


# ---------------------------------------------------------------------------
# GPU: this package's K1 program.  The tests force the split count (an explicit ``splits`` / ``block_n``, or a
# verdict in ``K1_SPLIT_RULES`` with the SM-count floor lifted), so they exercise the split-K program on every device
# whatever the planner keeps on the deterministic program there, and assert by kernel name that it ran.
# ---------------------------------------------------------------------------

# K1 weight-gradient rows ``out[N, K] = G[T, N].T @ X[T, K]``: label -> (N, K); the thin rows of the GLM-5.2 table
# (indexer heads, a 48-wide row) and two wide rows.
K1_ROWS = {
    "indexer_hw": (32, 6144),
    "thin_48": (48, 5120),
    "indexer_k": (128, 6144),
    "kv_a": (576, 6144),
    "wide_512": (512, 4096),
}
K1_RAGGED_T = (16231, 129, 4097, 16172, 777, 65)


def _k1_wgrad_inputs(row, T, out_dtype, device, seed):
    N, K = K1_ROWS[row]
    g = torch.Generator(device=device).manual_seed(seed)
    G, X = _bf16((T, N), device, g), _bf16((T, K), device, g)
    out = torch.full((N, K), float("nan"), device=device, dtype=out_dtype)
    return G, X, out


def _k1_check(out, ref, what=""):
    """The K1 accuracy rule: the deterministic package's budgets against FP64, absolute (BF16 output: every element
    within ``1e-2 + 1e-2 * |ref|``; FP32 output: relative Frobenius error at most 5e-5 and largest absolute error at
    most 1e-3 of the largest reference magnitude).  The deterministic program's error on the same inputs is data, not
    part of the rule."""
    assert bool(torch.isfinite(out).all()), what
    if out.dtype == torch.bfloat16:
        bad = (out.double() - ref).abs() > BF16_ATOL + BF16_RTOL * ref.abs()
        assert int(bad.sum()) == 0, (what, int(bad.sum()))
    else:
        assert _rel_fro(out, ref) <= F32_REL_FRO_MAX, (what, _rel_fro(out, ref))
        assert float(
            (out.double() - ref).abs().max()
        ) <= F32_MAX_ABS_OVER_REF_ABSMAX * float(ref.abs().max()), what


def _k1_prepare(G, X, out, arch, splits, block_n=None):
    """This package's prepared weight gradient at an explicit split count (the transposed-store launch)."""
    block_n = block_n or cake_backend.k1_block_n(int(G.shape[1]))
    _require_template(
        arch, cake_backend.atomic_k1_template(block_n, out.dtype == torch.float32)
    )
    prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
        X.t(),
        G,
        out,
        transposed_out=True,
        splits=splits,
        block_n=block_n,
    )
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
    assert (prepared.splits, prepared.plan.block_n) == (splits, block_n)
    return prepared


def _force_k1_rule(monkeypatch, G, X, out, splits, block_n=None):
    """The force of :func:`k1_rule` through ``monkeypatch``; returns the template."""
    arch = det_backend.arch_for(out.device)
    N, K = int(G.shape[1]), int(X.shape[1])
    block_n = block_n or cake_backend.k1_block_n(N)
    f32 = out.dtype == torch.float32
    _require_template(arch, cake_backend.atomic_k1_template(block_n, f32))
    key = (arch, 1, K, N, "f32" if f32 else "bf16")
    monkeypatch.setitem(cake_backend.K1_MIN_SM_COUNT, arch, 0)
    monkeypatch.delitem(cake_backend.K1_RULE_MIN_SM_COUNT, key, raising=False)
    monkeypatch.delitem(cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS, key, raising=False)
    monkeypatch.delitem(
        cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS, key, raising=False
    )
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (block_n, splits))
    monkeypatch.setattr(cake_backend, "K1_SPLIT_OVERRIDES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RANGE_OVERRIDES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_ONE_SPLIT_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    return cake_backend.atomic_k1_template(block_n, f32)


@pytest.fixture
def no_k1_dsm(monkeypatch):
    """Take the cluster ranges of :data:`K1_DSM_RANGES_BY_SM` out of the planner for a test of the K1 tables it comes
    before (:func:`plan_atomic_gemm` reads it first; ``test_gpu_k1_dsm_ranges_run_at_their_edges`` holds it),
    with the convenience caches cleared before and after: no entry planned without the table outlives the test."""
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    monkeypatch.setattr(cake_backend, "K1_DSM_RANGES_BY_SM", {})
    yield
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()


@pytest.fixture
def k1_rule(monkeypatch):
    """``k1_rule(G, X, out, splits)``: a verdict for the row (and no SM-count floor, excluded token counts, chain or
    cluster ranges or tables of exactly one SM count), so the convenience path runs this package's program at that
    split count; returns the template."""
    yield functools.partial(_force_k1_rule, monkeypatch)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()


def _arena_tickets_zero():
    """Whether the K1 tickets and the deterministic program's slice counters of every arena hold zeros, live and
    retired by a growth (both zero between complete launches)."""
    for arena in cake_backend._ATOMIC_ARENAS.values():
        bufs = [arena.buffers.get("k1_tickets"), arena.buffers.get("k1_det_counters")]
        # an arena without a ``retired`` list (its growths release the buffer they replace): its live buffers only
        bufs += [b for b in getattr(arena, "retired", ()) if b.dtype == torch.uint32]
        if any(b is not None and b.to(torch.int64).any() for b in bufs):
            return False
    return True


@pytest.mark.parametrize("splits", [2, 3])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("row", sorted(K1_ROWS))
def test_gpu_k1_budgets_over_reruns(row, out_dtype, splits, k1_rule):
    """The split-K weight gradient meets the deterministic budgets against FP64 on every one of ``RERUNS`` prepared
    launches and convenience calls, each into a NaN-poisoned ``out``; its distance to the deterministic result stays
    within the same budget; two splits repeat bit for bit; the tickets are zero after every launch; the profiler
    shows this package's program and no deterministic K1 program."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs(row, 16231, out_dtype, device, SEED)
    ref = G.double().t() @ X.double()
    det_out = det_backend.cake_dense_gemm_det_train_projection_wgrad(
        G, X, torch.empty_like(out)
    )
    prepared = _k1_prepare(G, X, out, arch, splits)
    _assert_ran(prepared.launch, prepared.module_name)
    k1_rule(G, X, out, splits)
    first = None
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, ref, f"prepared rerun {i}")
        _k1_check(out, det_out.double(), f"vs det rerun {i}")
        assert prepared.counters_are_zero(), i
        if splits == 2:
            first = out.clone() if first is None else first
            assert torch.equal(out, first), i
        out.fill_(float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
        torch.cuda.synchronize()
        _k1_check(out, ref, f"convenience rerun {i}")
    assert _arena_tickets_zero()
    _assert_ran(
        lambda: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out),
        prepared.module_name,
    )


@pytest.mark.parametrize("splits", [1, 2, 3, 4, 8])
def test_gpu_k1_explicit_split_counts(splits):
    """Every explicit split count (``1``: this package's program storing whole tiles) gives the budgeted result and
    leaves the tickets zero."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 1)
    prepared = _k1_prepare(G, X, out, arch, splits)
    for _ in range(3):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double())
        assert prepared.counters_are_zero()


# The 192-wide row-major program (sm_107a): ``(d_out, d_in, T, splits)``, the 4096 -> 1024 row it serves (22 column
# tiles of ``d_in``, the last 64 columns wide: its first CTA's third 32-column B box and its second CTA lie past ``N``)
# and a ragged one (``d_in`` 4104 cuts a B box, ``d_out`` 1032 a pair tile; two splits).
K1_ROW_MAJOR_ROWS = ((1024, 4096, 16231, 1), (1032, 4104, 1001, 2))


@pytest.mark.parametrize("transposed_out", [True, False])
@pytest.mark.parametrize(
    "shape", K1_ROW_MAJOR_ROWS, ids=lambda s: "x".join(map(str, s))
)
def test_gpu_k1_row_major_program(shape, transposed_out, k1_rule):
    """The row-major-store program (operands exchanged) meets the FP32 budget against FP64 for either orientation of
    the caller's views, prepared and through the convenience path, each launch into a NaN-poisoned ``out``; one split
    repeats bit for bit and stores what the deterministic budget allows; the tickets are zero after every launch; the
    profiler shows this package's program."""
    device, arch = _atomic_device_or_skip()
    if arch != "sm_107a":
        pytest.skip("the 192-wide row-major program ships on sm_107a only")
    template = cake_backend.atomic_k1_template(192, True)
    _require_template(arch, template)
    d_out, d_in, T, splits = shape
    g = torch.Generator(device=device).manual_seed(SEED + 7)
    G, X = _bf16((T, d_out), device, g), _bf16((T, d_in), device, g)
    out = torch.full((d_out, d_in), float("nan"), device=device, dtype=torch.float32)
    ref = G.double().t() @ X.double()
    A, B = (X.t(), G) if transposed_out else (G.t(), X)
    prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
        A,
        B,
        out,
        transposed_out=transposed_out,
        splits=splits,
        block_n=192,
    )
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
    assert prepared.template == template and not prepared.plan.out_t
    assert prepared.plan.swapped == transposed_out
    _assert_ran(prepared.launch, prepared.module_name)
    first = None
    for i in range(3):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, ref, f"prepared rerun {i}")
        assert prepared.counters_are_zero(), i
        if splits == 1:
            first = out.clone() if first is None else first
            assert torch.equal(out, first), i
    if transposed_out:
        assert k1_rule(G, X, out, splits, block_n=192) == template
        for i in range(2):
            out.fill_(float("nan"))
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
            torch.cuda.synchronize()
            _k1_check(out, ref, f"convenience rerun {i}")
        assert _arena_tickets_zero()


# The one-split programs of three accumulation chains: ``(d_out, d_in, T, block_n)``.  961 tokens span 16 K blocks
# (chains of 5, 5 and 6), 4097 an odd 65, 193 four (1, 1 and 2); the 192-wide row-major-store program is sm_107a's.
K1_CHAIN_SHAPES = (
    (512, 4096, 961, 128),
    (512, 4096, 4097, 128),
    (512, 4096, 193, 128),
    (1024, 5120, 16231, 256),
    (1024, 4096, 4097, 192),
)


@pytest.mark.parametrize("shape", K1_CHAIN_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_gpu_k1_chain_programs(shape, monkeypatch):
    """The one-split program of three accumulation chains, prepared with ``chains=3``, meets the K1 accuracy rule (the
    deterministic package's budget against FP64, :func:`_k1_check`); each launch into a NaN-poisoned ``out`` repeats bit
    for bit, also replayed from a CUDA graph; the convenience path at a token count of a range of
    :data:`K1_CHAIN_RANGES_BY_SM` for the device's SM count stores the same bits; the profiler shows the three-chain
    program; fewer than three K blocks raise ``ValueError``."""
    device, arch = _atomic_device_or_skip()
    d_out, d_in, T, block_n = shape
    if block_n == 192 and arch != "sm_107a":
        pytest.skip("the 192-wide row-major program ships on sm_107a only")
    template = cake_backend.atomic_k1_template(block_n, True, 3)
    _require_template(arch, template)
    g = torch.Generator(device=device).manual_seed(SEED + 11)
    G, X = _bf16((T, d_out), device, g), _bf16((T, d_in), device, g)
    out = torch.full((d_out, d_in), float("nan"), device=device, dtype=torch.float32)
    prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
        X.t(),
        G,
        out,
        transposed_out=True,
        block_n=block_n,
        chains=3,
    )
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
    assert prepared.template == template
    assert (prepared.splits, prepared.plan.chains) == (1, 3)
    _assert_ran(prepared.launch, prepared.module_name)
    out.fill_(float("nan"))
    prepared.launch()
    torch.cuda.synchronize()
    _k1_check(out, G.double().t() @ X.double(), "prepared")
    first = out.clone()
    for i in range(3):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert torch.equal(out, first), i
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.launch()
    for i in range(2):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, first), ("graph", i)
    # the convenience path at a token count of a chain range for this device
    sm_count = det_backend.device_facts(device).sm_count
    key = (arch, 1, d_in, d_out, "f32")
    monkeypatch.setattr(
        cake_backend,
        "K1_CHAIN_RANGES_BY_SM",
        {(arch, sm_count): {key: ((T, T, (block_n, 3)),)}},
    )
    monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
    # the stream-K rows and the ranges of K1_ORDERED_RANGES_BY_SM and K1_PARITY_RANGES_BY_SM come before the chain
    # ranges
    monkeypatch.setattr(cake_backend, "K1_WGRAD_SK_RULES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_ORDERED_RANGES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    try:
        out.fill_(float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
        torch.cuda.synchronize()
        assert torch.equal(out, first)
        _assert_ran(
            lambda: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
                G, X, out
            ),
            prepared.module_name,
        )
    finally:
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    short = torch.zeros(128, d_out, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="at one split"):
        cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            torch.zeros(128, d_in, device=device, dtype=torch.bfloat16).t(),
            short,
            torch.empty(d_out, d_in, device=device),
            transposed_out=True,
            block_n=block_n,
            chains=3,
        )


def test_gpu_k1_batched_mla_head_views():
    """The batched MLA weight gradient (64 heads, strided head views of ``[T, H, slot]`` activations): this
    package's split-K program at an explicit split count (one slot per head tile and split)."""
    device, arch = _atomic_device_or_skip()
    _require_template(arch, cake_backend.atomic_k1_template(256, False))
    g = torch.Generator(device=device).manual_seed(SEED + 2)
    T, H, slot, D_in, D_out = 2048, 64, 256, 192, 512
    act = _bf16((T, H, slot), device, g)[..., :D_in]
    d_out = _bf16((T, H, D_out), device, g)
    A, B = (
        act.permute(1, 2, 0),
        d_out.permute(1, 0, 2),
    )  # [H, D_in, T] (unit stride on m), [H, T, D_out]
    out = torch.full(
        (H, D_in, D_out), float("nan"), device=device, dtype=torch.bfloat16
    )
    ref = A.double() @ B.double()
    prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
        A, B, out, splits=2, block_n=256
    )
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
    assert (prepared.plan.L, prepared.splits, prepared.plan.swapped) == (H, 2, True)
    for _ in range(3):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, ref)
        assert prepared.counters_are_zero()
    # the planner keeps the batched rows on the deterministic program: the default convenience path matches it
    out2 = torch.empty_like(out)
    cake_backend.cake_dense_gemm_nondet_train_projection(A, B, out2)
    torch.cuda.synchronize()
    _k1_check(out2, ref)


def test_gpu_k1_ragged_token_counts_reuse_poisoned_scratch(k1_rule):
    """Changing ``T`` up and down on one stream with a verdict for the row: the convenience path reuses its arena
    (grown, never shrunk); the split slots are poisoned with NaN before each call (a stale value never reaches
    ``out``: the padding columns the last split reads are discarded), every output -- poisoned too -- meets the
    budget, and the tickets are zero throughout.  The shortest token count (65) is below the split count's 129 rows,
    so that call runs the deterministic program (its bits)."""
    device, arch = _atomic_device_or_skip()
    k1_rule(*_k1_wgrad_inputs("thin_48", 128, torch.bfloat16, device, SEED), 3)
    for i, T in enumerate(K1_RAGGED_T):
        G, X, out = _k1_wgrad_inputs(
            "thin_48", T, torch.bfloat16, device, SEED + 10 + i
        )
        for arena in cake_backend._ATOMIC_ARENAS.values():
            if "k1_slots" in arena.buffers:
                arena.buffers["k1_slots"].fill_(float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double(), f"T={T}")
        assert _arena_tickets_zero(), T
        if not cake_backend.k1_splits_fit(3, T):
            assert torch.equal(
                out, det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X)
            )
    assert len(cake_backend._ATOMIC_ARENAS) == 1
    (arena,) = cake_backend._ATOMIC_ARENAS.values()
    assert arena.version >= 2  # created, then grown or reused; never shrunk


def test_gpu_k1_small_token_counts_guard():
    """An explicit split count needs ``splits * 64 <= T + 63``: below, ``ValueError`` (the program would read an
    empty K range); the planner's rows fall back to the deterministic program instead (CPU test)."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_hw", 64, torch.bfloat16, device, SEED + 3)
    _k1_prepare(G, X, out, arch, 1)
    with pytest.raises(ValueError, match="splits must be in"):
        _k1_prepare(G, X, out, arch, 2)


def test_gpu_torch_deterministic_mode_alerts_before_an_order_varying_launch(
    k1_rule, torch_determinism
):
    """Under ``torch.use_deterministic_algorithms(True)`` a call whose launch is fixed order
    (two splits: one add) runs without an alert and repeats bit for bit; a launch whose result varies run to run
    (three splits) raises PyTorch's ``RuntimeError`` before it launches (``out`` untouched), or with
    ``warn_only=True`` warns and runs this package's program within the accuracy rule.  The setting never changes
    the program that runs."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 40)
    # the FP64 reference before the mode is on (its cuBLAS GEMM would need CUBLAS_WORKSPACE_CONFIG under it)
    ref = G.double().t() @ X.double()
    k1_rule(G, X, out, 2)
    torch.use_deterministic_algorithms(True)
    with _no_warnings():
        first = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
            G, X, out
        ).clone()
        for _ in range(RERUNS):
            again = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
                G, X, out
            )
            assert torch.equal(again, first)
    k1_rule(G, X, out, 3)
    out.fill_(7.0)
    with pytest.raises(RuntimeError, match="at 3 splits"):
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
    torch.cuda.synchronize()
    assert bool((out == 7.0).all())
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="at 3 splits"):
        got = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
    torch.cuda.synchronize()
    torch.use_deterministic_algorithms(False)
    _k1_check(first, ref, "two splits")
    _k1_check(got, ref, "three splits")
    assert _arena_tickets_zero()


#: The cluster program of the shipped row ``4096 -> 64`` FP32 at 2048 tokens on each part of
#: :data:`K1_DSM_RANGES_BY_SM`: ``(block_n, splits)``.
K1_DSM_ALERT_ROWS = {
    ("sm_100a", 148): (32, 2),
    ("sm_100a", 152): (64, 2),
    ("sm_103a", 152): (64, 2),
}


def test_gpu_the_cluster_program_does_not_alert(torch_determinism):
    """The shipped cluster row ``4096 -> 64`` FP32 at 2048 tokens of :data:`K1_DSM_RANGES_BY_SM` (the cluster program
    of :data:`K1_DSM_ALERT_ROWS` on the part: two or three splits, added in cluster-rank order) launches without an
    alert or a warning under ``torch.use_deterministic_algorithms(True)``, and with ``warn_only=True``: the planning
    call, the call from the cached key and the prepared launch run the cluster program (by kernel name), every output
    meets the deterministic package's FP32 budget against FP64 (:func:`_k1_check`), and all of them have the same
    bits.  ``test_the_cluster_program_does_not_alert`` checks the same calls on the CPU binding.  A part of any other SM
    count or architecture skips."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    if (arch, sm_count) not in K1_DSM_ALERT_ROWS:
        pytest.skip(f"no K1 cluster row is shipped for {arch} on {sm_count} SMs")
    T, N, M = 2048, 64, 4096
    dsm = cake_backend.k1_dsm_plan(
        arch=arch, L=1, M=M, N=N, K=T, out_f32=True, sm_count=sm_count
    )
    assert dsm == K1_DSM_ALERT_ROWS[(arch, sm_count)]
    _require_template(arch, cake_backend.dsm_k1_template(*dsm))
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    g = torch.Generator(device=device).manual_seed(SEED + 99)
    G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
    # the FP64 reference before the mode is on (its cuBLAS GEMM would need CUBLAS_WORKSPACE_CONFIG under it)
    ref = G.double().t() @ X.double()
    out = torch.full((N, M), float("nan"), device=device, dtype=torch.float32)
    got = []
    torch.use_deterministic_algorithms(True)
    with _no_warnings():
        for _ in range(2):  # the planning call, then the call from the cached key
            out.fill_(float("nan"))
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            got.append(out.clone())
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
            G, X, out
        )
        out.fill_(float("nan"))
        assert prepared.launch() is out
        got.append(out.clone())
        torch.cuda.synchronize()
    ((_, entry),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert entry.plan.dsm and entry.arena is None
    assert (entry.plan.block_n, entry.plan.splits) == dsm and not entry.order_varying
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
    assert prepared.plan.dsm and prepared.plan.reproducible
    assert (prepared.plan.block_n, prepared.plan.splits) == dsm
    # still under the mode: the calls ran the cluster program
    _assert_ran(
        lambda: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out),
        entry.module_name,
    )
    _assert_ran(prepared.launch, prepared.module_name)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with _no_warnings():
        out.fill_(float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
        got.append(out.clone())
        out.fill_(float("nan"))
        prepared.launch()
        got.append(out.clone())
        torch.cuda.synchronize()
    torch.use_deterministic_algorithms(False)
    for i, result in enumerate(got):
        _k1_check(result, ref, f"cluster program call {i}")
        assert torch.equal(result, got[0]), f"cluster program call {i}: other bits"
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()


def test_gpu_det_then_atomic_share_a_stream_not_a_workspace(k1_rule):
    """A deterministic call and a call of this package of one geometry on one stream: each package keeps its own
    arena (the deterministic package's slabs never reach this package's tickets)."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 50)
    k1_rule(G, X, out, 2)
    ref = G.double().t() @ X.double()
    for _ in range(3):
        det = det_backend.cake_dense_gemm_det_train_projection_wgrad(
            G, X, torch.empty_like(out)
        )
        atomic = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
        torch.cuda.synchronize()
        _k1_check(det, ref)
        _k1_check(atomic, ref)
    ours = {
        id(b) for a in cake_backend._ATOMIC_ARENAS.values() for b in a.buffers.values()
    }
    theirs = {id(b) for a in det_backend._ARENAS.values() for b in a.buffers.values()}
    assert ours and not ours & theirs
    assert _arena_tickets_zero()


def test_gpu_k1_deterministic_program_through_the_binding():
    """K1 calls of this package that run the deterministic program -- the forward, the input gradient (BF16
    and FP32 out), the weight gradient of rows without a verdict (both orientations, BF16 and FP32 out, another
    ``sk``), the batched head views of the generic entry -- go through this package's binding and give the
    deterministic package's bits: dense and padded outputs (the padding untouched), fresh tensors on every call, a
    larger row growing the arena (the smaller rows' bindings carved again), another stream with an arena of its own,
    the slice counters zero after the launches; a batched call the deterministic package runs on its swapped
    small-token program stays unbound (that package's call); a binding the deterministic package refuses raises its
    error."""
    device = _det_device_or_skip()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    rows = (("kv_a", 576, 6144), ("4096to128", 128, 4096), ("wide", 2048, 4096))
    ran, swapped = set(), set()

    def check(T, seed, pad_out=False):
        g = torch.Generator(device=device).manual_seed(seed)
        for name, N, K in rows:
            X, W, G = (
                _bf16((T, K), device, g),
                _bf16((N, K), device, g, K**-0.5),
                _bf16((T, N), device, g),
            )
            # op -> (out shape, out dtype, the deterministic package's call, this package's call)
            # fmt: off
            calls = {
                "fwd": ((T, N), torch.bfloat16, lambda o: det_backend.cake_dense_gemm_det_train_projection_forward(X, W, o),
                        lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, o)),
                "dgrad": ((T, K), torch.bfloat16, lambda o: det_backend.cake_dense_gemm_det_train_projection_dgrad(G, W, o),
                          lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, o)),
                "dgrad_f32": ((T, K), torch.float32, lambda o: det_backend.cake_dense_gemm_det_train_projection_dgrad(G, W, o),
                              lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, o)),
                "wgrad": ((N, K), torch.bfloat16, lambda o: det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X, o),
                          lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, o)),
                "wgrad_f32": ((N, K), torch.float32, lambda o: det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X, o),
                              lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, o)),
                "wgrad_sk": ((N, K), torch.float32, lambda o: det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X, o, sk=False),
                             lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, o, sk=False)),
            }
            # fmt: on
            arch = det_backend.arch_for(device)
            sm_count = det_backend.device_facts(device).sm_count
            for op, ((rows_, cols), dtype, det_call, call) in calls.items():
                f32 = dtype == torch.float32
                kw = dict(arch=arch, L=1, M=K, N=N, K=T, out_f32=f32, sm_count=sm_count)
                if op.startswith("wgrad") and (
                    cake_backend.k1_plan(**kw) or cake_backend.k1_dsm_plan(**kw)
                ):
                    continue  # a row with a verdict or a cluster range: this package's program
                if pad_out:
                    storage = torch.full(
                        (rows_, cols + 64), 7.0, device=device, dtype=dtype
                    )
                    out = storage[:, :cols]
                else:
                    out = torch.empty(rows_, cols, device=device, dtype=dtype)
                want = det_call(out).clone()
                for _ in range(2):
                    out.fill_(float("nan"))
                    assert call(out) is out
                    assert torch.equal(out, want), (name, op, T, pad_out)
                if pad_out:
                    assert bool((storage[:, cols:] == 7.0).all()), (name, op)
                ran.add(op)
        # the batched MLA forward / input gradient head views of the generic entry
        H, slot, D_in, D_out = 64, 256, 192, 512
        act = _bf16((T, H, slot), device, g)[..., :D_in]
        weight = _bf16((H, D_in, D_out), device, g, D_in**-0.5)
        d_out = _bf16((T, H, D_out), device, g)
        for A, B, shape in (
            (act.permute(1, 0, 2), weight, (T, H, D_out)),
            (d_out.permute(1, 0, 2), weight.transpose(1, 2), (T, H, D_in)),
        ):
            out = torch.empty(shape, device=device, dtype=torch.bfloat16).permute(
                1, 0, 2
            )
            if det_backend._swap_ab_launch(A, B, out, False) is not None:
                swapped.add((T, shape))
            want = det_backend.cake_dense_gemm_det_train_projection(A, B, out).clone()
            out.fill_(float("nan"))
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection(A, B, out) is out
            )
            assert torch.equal(out, want), (T, tuple(shape))

    for T, pad_out in ((1000, False), (4097, True), (64, False)):
        check(T, SEED + 80 + T, pad_out)
    assert {"fwd", "dgrad", "dgrad_f32", "wgrad_f32", "wgrad_sk"} <= ran
    det_entries = [e for e in cake_backend._K1_DET_CACHE.values() if e.k1 is not None]
    unbound = [e for e in cake_backend._K1_DET_CACHE.values() if e.k1 is None]
    assert len(unbound) == len(swapped) and all(e.plan is None for e in unbound)
    wgrad_entries = [
        e for e in cake_backend._K1_ATOMIC_CACHE.values() if e.plan is None
    ]
    assert det_entries and wgrad_entries
    assert all(e.k1 is not None for e in wgrad_entries)
    assert all(e.det_plan is not None for e in wgrad_entries)
    versions = {id(e): e.k1.version for e in det_entries + wgrad_entries}
    # the arena grows: the bindings are carved again
    for arena in list(cake_backend._ATOMIC_ARENAS.values()):
        for kind in ("k1_det_ws", "k1_det_counters"):
            if kind in arena.buffers:
                arena.require(kind, 2 * arena.buffers[kind].numel())
    for T, pad_out in ((1000, False), (4097, True), (64, False)):
        check(T, SEED + 81 + T, pad_out)
    assert all(e.k1.version != versions[id(e)] for e in det_entries + wgrad_entries)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        check(1000, SEED + 82)
    torch.cuda.synchronize()
    assert stream.cuda_stream in {key[1] for key in cake_backend._ATOMIC_ARENAS}
    for arena in cake_backend._ATOMIC_ARENAS.values():
        counters = arena.buffers.get("k1_det_counters")
        assert counters is not None and not bool(counters.view(torch.int32).any())
    # a binding the deterministic package refuses: its error
    g = torch.Generator(device=device).manual_seed(SEED + 83)
    X, W = _bf16((1001, 4096), device, g), _bf16((128, 4096), device, g)
    storage = torch.empty(1001 * 4096 + 8, device=device, dtype=torch.bfloat16)
    Xm = storage[1 : 1 + 1001 * 4096].view(1001, 4096)
    Xm.copy_(X)
    with pytest.raises(ValueError, match="16-byte aligned") as refused:
        det_backend.cake_dense_gemm_det_train_projection_forward(Xm, W)
    with pytest.raises(ValueError, match=re.escape(str(refused.value))):
        cake_backend.cake_dense_gemm_nondet_train_projection_forward(Xm, W)


def test_gpu_k1_bound_launch_is_the_unbound_launch(k1_rule):
    """A row with a verdict at two splits (two partials per element: bitwise reproducible): the convenience path's
    bound launch gives the bits of the per-call ordered launch (``_launch_atomic_k1_unbound``) and of the prepared
    launch, on fresh tensors every call, the tickets zero after."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 4097, torch.float32, device, SEED + 90)
    k1_rule(G, X, out, 2)
    prepared = _k1_prepare(G, X, torch.empty_like(out), arch, 2)
    prepared.launch()
    want = prepared.out.clone()
    for i in range(3):
        G2, X2, out2 = G.clone(), X.clone(), torch.full_like(out, float("nan"))
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G2, X2, out2)
            is out2
        )
        torch.cuda.synchronize()
        assert torch.equal(out2, want), i
        assert _arena_tickets_zero()
    entries = [e for e in cake_backend._K1_ATOMIC_CACHE.values() if e.plan is not None]
    (entry,) = entries
    assert entry.k1 is not None and entry.plan.splits == 2
    det = det_backend
    index = out.get_device()
    unbound = torch.full_like(out, float("nan"))
    cake_backend._launch_atomic_k1_unbound(
        entry,
        X.t(),
        G,
        unbound,
        index,
        det._current_raw_stream(index),
        int(det.pdl_enabled(None, index)),
    )
    torch.cuda.synchronize()
    assert torch.equal(unbound, want) and _arena_tickets_zero()


@pytest.mark.parametrize("splits", [1, 2])
def test_gpu_k1_pdl_waits_for_an_early_producer(splits, k1_rule):
    """This package's weight gradient as a programmatic dependent launch: the deterministic forward before it
    (K = 8192) triggers its dependents at entry and writes ``G``; the weight gradient starts while it runs and must
    still read every row it wrote, and its tickets end zero for the next launch on the stream."""
    device, arch = _atomic_device_or_skip()
    N, K = K1_ROWS["indexer_k"]
    T = 4096
    g = torch.Generator(device=device).manual_seed(SEED + 60)
    H = _bf16((T, 8192), device, g)
    Wg = (torch.randn(N, 8192, device=device, generator=g) * 8192**-0.5).to(
        torch.bfloat16
    )
    X = _bf16((T, K), device, g)
    G = torch.empty(T, N, device=device, dtype=torch.bfloat16)
    out = torch.empty(N, K, device=device, dtype=torch.bfloat16)
    k1_rule(G, X, out, splits)
    for _ in range(3):
        G.fill_(float("nan"))
        out.fill_(float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_forward(
            H, Wg, G, enable_pdl=True
        )
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
            G, X, out, enable_pdl=True
        )
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double())
    assert _arena_tickets_zero()


def test_gpu_k1_bound_deterministic_program_waits_for_an_early_producer():
    """The deterministic program through this package's binding as a programmatic dependent launch: the
    deterministic forward before it (K = 8192) triggers its dependents at entry and writes ``G``; the input gradient
    (bound, ``enable_pdl=True``) starts while it runs and must still read every row it
    wrote -- the deterministic package's bits -- and its slice counters end zero."""
    device = _det_device_or_skip()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    T, N, K = 4096, 128, 4096
    g = torch.Generator(device=device).manual_seed(SEED + 61)
    H = _bf16((T, 8192), device, g)
    Wg = _bf16((N, 8192), device, g, 8192**-0.5)
    W = _bf16((N, K), device, g, K**-0.5)
    G = torch.empty(T, N, device=device, dtype=torch.bfloat16)
    dX = torch.empty(T, K, device=device, dtype=torch.bfloat16)
    det_backend.cake_dense_gemm_det_train_projection_forward(H, Wg, G, enable_pdl=False)
    want = det_backend.cake_dense_gemm_det_train_projection_dgrad(
        G, W, enable_pdl=False
    ).clone()
    for _ in range(3):
        G.fill_(float("nan"))
        dX.fill_(float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_forward(
            H, Wg, G, enable_pdl=True
        )
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(
                G, W, dX, enable_pdl=True
            )
            is dX
        )
        torch.cuda.synchronize()
        assert torch.equal(dX, want)
    entries = [e for e in cake_backend._K1_DET_CACHE.values() if e.k1 is not None]
    assert entries
    for arena in cake_backend._ATOMIC_ARENAS.values():
        counters = arena.buffers.get("k1_det_counters")
        assert counters is None or not bool(counters.view(torch.int32).any())


# ---------------------------------------------------------------------------
# GPU: this package's K2 program
# ---------------------------------------------------------------------------

ROUTER_T = 4099


def _router_inputs(T, device, seed, K=6144):
    g = torch.Generator(device=device).manual_seed(seed)
    N = 256
    X = torch.randn(T, K, device=device, generator=g)
    W = torch.randn(N, K, device=device, generator=g) * 0.02
    G = torch.randn(T, N, device=device, generator=g) * 1e-3
    return X, W, G


def _router_case(op, X, W, G):
    """``(A, B, out)`` of the router GEMM of ``op`` (``fwd``: ``X @ W.T``, ``dgrad``: ``G @ W``, ``wgrad``:
    ``G.T @ X``)."""
    if op == "fwd":
        return (
            X,
            W.t(),
            torch.full((X.shape[0], W.shape[0]), float("nan"), device=X.device),
        )
    if op == "dgrad":
        return G, W, torch.full((G.shape[0], W.shape[1]), float("nan"), device=X.device)
    return G.t(), X, torch.full((G.shape[1], X.shape[1]), float("nan"), device=X.device)


def _force_router_rule(monkeypatch, op, A, B, spec):
    """The force of :func:`router_rule` through ``monkeypatch``."""
    arch = det_backend.arch_for(A.device)
    if op == "fwd":
        key = (arch, "kk", int(A.shape[1]), int(B.shape[1]))
    elif op == "dgrad":
        key = (arch, "kn", int(B.shape[1]), int(A.shape[1]))
    else:
        key = (arch, "nn_t", int(B.shape[1]), int(A.shape[0]))
    monkeypatch.setitem(cake_backend.ROUTER_RULES, key, spec)
    # a forced verdict applies on this device whatever its SM count (also on a part with a table of its own)
    sm_count = det_backend.device_facts(A.device).sm_count
    monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
    tables = dict(cake_backend.ROUTER_RULES_BY_SM)
    tables.pop((arch, sm_count), None)
    monkeypatch.setattr(cake_backend, "ROUTER_RULES_BY_SM", tables)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()


@pytest.fixture
def router_rule(monkeypatch):
    """``router_rule(A, B, spec)``: a verdict for the router row of ``(A, B)``, so the default plan runs this
    package's program with the partition ``spec`` (or the deterministic program with ``None``)."""
    yield functools.partial(_force_router_rule, monkeypatch)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()


def _router_prepare(A, B, out, splits, arch, spec=None):
    _require_template(
        arch,
        "router_fp32_gemm_atomic_"
        + ("nn_t" if A.stride(0) == 1 else "kk")
        + (
            cake_backend.ROUTER_HEAD_TAIL_SUFFIX
            if cake_backend.router_head_tail(spec)
            else ""
        ),
    )
    prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
        A, B, out, splits=splits
    )
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
    return prepared


def _router_segment_reference(A, B, plan):
    """``(expected, bound)`` of a launch of ``plan``: each tile's segments' partials -- the deterministic program's
    one-split store of the segment's K range, which this package's program reproduces bit for bit -- summed in K
    order, and the bound of a reordering of that sum (``0`` for at most two partials: either order rounds alike)."""
    K = int(A.shape[1])
    partials = {}
    per_tile = _segments(
        plan.m_pairs,
        plan.n_tiles,
        plan.k_iters,
        plan.units,
        plan.head_spans,
        plan.head_steps,
    )
    for k0, k1 in sorted({seg for segs in per_tile for seg in segs}):
        lo, hi = (
            k0 * cake_backend.ROUTER_BLOCK_K,
            min(K, k1 * cake_backend.ROUTER_BLOCK_K),
        )
        part = torch.empty(int(A.shape[0]), int(B.shape[1]), device=A.device)
        det_backend.cake_dense_gemm_det_train_router(
            A[:, lo:hi], B[lo:hi, :], part, splits=1
        )
        partials[(k0, k1)] = part.t() if plan.swapped else part  # the launch's [M, N]
    expected = torch.empty(plan.M, plan.N, device=A.device)
    bound = torch.zeros(plan.M, plan.N, device=A.device, dtype=torch.float64)
    bm, bn = 2 * cake_backend.ROUTER_BLOCK_M, cake_backend.ROUTER_BLOCK_N
    for t, segs in enumerate(per_tile):
        pair, col = divmod(t, plan.n_tiles)
        rows = slice(pair * bm, min(plan.M, (pair + 1) * bm))
        cols = slice(col * bn, min(plan.N, (col + 1) * bn))
        if rows.start >= plan.M:
            continue
        tile = [partials[s][rows, cols] for s in segs]
        acc = tile[0].clone()
        for p in tile[1:]:
            acc = acc + p
        expected[rows, cols] = acc
        if len(tile) > 2:
            mag = sum(p.double().abs() for p in tile)
            bound[rows, cols] = (len(tile) - 1) * 2.0**-23 * mag
    return expected, bound


@pytest.mark.parametrize(
    "op, splits, spec",
    [
        ("fwd", None, "forward"),
        ("fwd", 1, None),
        ("fwd", 2, None),
        ("wgrad", None, "pairs"),
        ("wgrad", None, "H0.5x3"),
        ("wgrad", None, "H0.5x4"),
        ("wgrad", None, "H1x4p44"),
        ("wgrad", None, "H1x6p44"),
        ("wgrad", 1, None),
        ("wgrad", 4, None),
        ("wgrad", 11, None),
    ],
)
def test_gpu_router_d11_gate(op, splits, spec, router_rule):
    """The router's accuracy gate, the deterministic package's shipped budget: ``rel_fro <= 2e-6`` against FP64 of
    the exact fp32 operands, finite, worst over ``RERUNS`` launches into a NaN-poisoned ``out``; the forward's top-8
    agrees with FP64 on every row and repeats bit for bit.  Against the deterministic program's one-split partials
    of each segment: a whole tile is their bits, two partials their sum (either order rounds alike), more within the
    reordering bound of their K-order sum.  One unit per tile (``splits=1``) is the deterministic program's
    one-split result.  The profiler shows this package's program."""
    _router_gate(op, splits, spec, router_rule, ROUTER_T, 6144, SEED + 20)


@pytest.mark.parametrize("T", [1000, 16172, 16231, 16384])
@pytest.mark.parametrize("spec", ["H1x4p44", "H1x5p44", "H1x4p80"])
def test_gpu_router_head_units_at_hidden_4096(spec, T, router_rule):
    """The head/tail partitions with fewer head units than CTA pairs (``"H<f>x<k>p<h>"``: the hidden 4096 verdict of
    ``sm_103a`` and the hidden 6144 verdict of 152-SM ``sm_103a`` parts; 80 head units are more than a 152-SM part's CTA pairs, so there the partition is
    ``"H1x4"``'s) at hidden 4096, 256 experts, the measured token counts and a short one: the gate of
    :func:`test_gpu_router_d11_gate`, on the partition :func:`cake_backend.router_partition_of` gives the device."""
    plan = _router_gate("wgrad", None, spec, router_rule, T, 4096, SEED + 24)
    sm_count = det_backend.device_facts(torch.device("cuda")).sm_count
    m_pairs = plan.m_tiles // 2
    partition = cake_backend.router_partition_of(
        spec, m_pairs, plan.n_tiles, plan.k_iters, sm_count
    )
    assert (plan.units, plan.head_spans, plan.head_steps) == partition
    if cake_backend.router_head_pairs(spec) >= sm_count // 2:
        assert partition == cake_backend.router_partition_of(
            spec.split("p")[0], m_pairs, plan.n_tiles, plan.k_iters, sm_count
        )
    else:
        assert plan.head_spans == cake_backend.router_head_pairs(spec) // plan.n_tiles


def _router_gate(op, splits, spec, router_rule, T, hidden, seed):
    """:func:`test_gpu_router_d11_gate` on ``T`` tokens of hidden size ``hidden``; returns the launch's plan."""
    device, arch = _atomic_device_or_skip()
    X, W, G = _router_inputs(T, device, seed, K=hidden)
    A, B, out = _router_case(op, X, W, G)
    if spec is not None:
        router_rule(op, A, B, spec)
    prepared = _router_prepare(A, B, out, splits, arch, spec)
    plan = prepared.plan
    assert bool(plan.head_spans) == bool(cake_backend.router_head_tail(spec))
    if op == "fwd":
        assert plan.reproducible
    ref = A.double() @ B.double()
    expected, bound = _router_segment_reference(A, B, plan)
    if plan.swapped:
        expected, bound = expected.t(), bound.t()
    _assert_ran(prepared.launch, prepared.module_name)
    first, worst = None, 0.0
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all()), i
        worst = max(worst, _rel_fro(out, ref))
        assert bool(((out.double() - expected.double()).abs() <= bound).all()), i
        if plan.reproducible:
            first = out.clone() if first is None else first
            assert torch.equal(out, first), i
        if op == "fwd" and i == 0:
            ref_top = torch.topk(ref, 8, dim=1).indices.sort(dim=1).values
            top = torch.topk(out.double(), 8, dim=1).indices.sort(dim=1).values
            assert int((top != ref_top).any(dim=1).sum()) == 0
    assert worst <= ROUTER_REL_FRO_MAX, worst
    if splits == 1:
        det_out = torch.empty_like(out)
        det_backend.cake_dense_gemm_det_train_router(A, B, det_out, splits=1)
        assert torch.equal(out, det_out)
    return plan


def _top8_rows_differ(out, ref, tie_ulps=16):
    """``(differing, ties)``: the rows whose top-8 index set differs from FP64's, outside the near ties -- rows whose
    FP64 8th and 9th values lie within ``tie_ulps`` FP32 ulps of the 8th, where any order of the FP32 sum may pick
    either -- and the count of those near ties."""
    top = torch.topk(ref, 9, dim=1).values
    gap = top[:, 7] - top[:, 8]
    tie = gap <= tie_ulps * torch.finfo(torch.float32).eps * top[:, 7].abs()
    ref_top = torch.topk(ref, 8, dim=1).indices.sort(dim=1).values
    got_top = torch.topk(out.double(), 8, dim=1).indices.sort(dim=1).values
    differ = (got_top != ref_top).any(dim=1) & ~tie
    return int(differ.sum()), int(tie.sum())


@pytest.mark.parametrize(
    "hidden, T",
    [
        (4096, 20480),
        (4096, 49152),
        (5120, 16172),
        (5120, 16231),
        (5120, 16384),
        (5120, 32768),
        (6144, 16385),
        (6144, 20480),
    ],
)
def test_gpu_router_tmem_head_tail_program_in_its_windows(hidden, T):
    """On a part whose router table runs a head/tail window with the head/tail program with the A parts in tensor
    memory (152-SM sm_103a: hidden 4096 at 20480 to 49152 tokens, hidden 5120 at 16172 to 32768 and hidden 6144 at
    16385 to 20480, ``ROUTER_HEAD_TAIL_TEMPLATES``; 212-SM sm_107a: hidden 5120 and 6144 at 16172 to 16384 tokens,
    ``ROUTER_HEAD_TAIL_TEMPLATES_BY_SM``) the default plan of the weight gradient (at the first and the last token
    count of each window and inside it) runs that program and meets the gate of :func:`test_gpu_router_d11_gate` over
    ``RERUNS`` launches into a NaN-poisoned ``out``: finite, within the reordering bound of the K-order sum of the
    deterministic program's one-split partials of its segments, relative Frobenius error against FP64 at most
    ``ROUTER_REL_FRO_MAX``.  On whole tiles (one unit per tile) it stores the bits of the three-chunk-buffer head/tail
    program at the same partition, which are the deterministic program's one-split result.  On a part of that
    architecture with another SM count, where the windows do not hold (``router_rules``: :data:`ROUTER_RULES` off
    :data:`ROUTER_SM_COUNT`, e.g. 148-SM sm_103a, or a table of its own with no window at that token count), the
    default plan is the deterministic program.  Skips on any other device and outside the part's windows."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    ts = cake_backend.router_head_tail_template(arch, sm_count)
    c3 = "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX
    if ts == c3:
        pytest.skip(f"{arch} with {sm_count} SMs runs {c3}")
    spec = cake_backend._router_rule_spec(
        "nn_t", arch=arch, sm_count=sm_count, M=hidden, N=256, K=T
    )
    home = cake_backend.ROUTER_SM_COUNT.get(arch, sm_count)
    home_spec = cake_backend._router_rule_spec(
        "nn_t", arch=arch, sm_count=home, M=hidden, N=256, K=T
    )
    on_table = cake_backend.router_head_tail(spec) is not None
    if not on_table and (
        spec is not None
        or home == sm_count
        or cake_backend.router_head_tail(home_spec) is None
    ):
        pytest.skip(
            f"no head/tail window of {arch} on {sm_count} SMs at {hidden}, T={T}"
        )
    X, W, G = _router_inputs(T, device, SEED + 30, K=hidden)
    A, B, out = _router_case("wgrad", X, W, G)
    if not on_table or not cake_backend.router_rules(arch, sm_count)[1]:
        plan, reason = cake_backend.plan_atomic_router(
            A, B, out, arch=arch, sm_count=sm_count
        )
        assert plan is None, plan
        assert reason == "the planner keeps the row on the deterministic program"
        prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(A, B, out)
        assert not isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
        return
    _require_template(arch, ts)
    _require_template(arch, c3)
    plan, reason = cake_backend.plan_atomic_router(
        A, B, out, arch=arch, sm_count=sm_count
    )
    assert plan is not None and plan.template == ts and plan.head_spans > 0, reason
    prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(A, B, out)
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
    assert prepared.plan == plan
    assert prepared.module_name == cake_jit.select_module(arch, ts)
    ref = A.double() @ B.double()
    expected, bound = _router_segment_reference(A, B, plan)
    expected, bound = expected.t(), bound.t()
    _assert_ran(prepared.launch, prepared.module_name)
    worst = 0.0
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all()), i
        worst = max(worst, _rel_fro(out, ref))
        assert bool(((out.double() - expected.double()).abs() <= bound).all()), i
    assert worst <= ROUTER_REL_FRO_MAX, worst
    # whole tiles: no part tile, no fill of out, one segment per tile
    whole = dataclasses.replace(plan, units=plan.tiles, head_spans=0, head_steps=0)
    assert not whole.part_tiles and whole.reproducible
    stores = {}
    for template in (ts, c3):
        stores[template] = torch.full_like(out, float("nan"))
        launch = cake_backend._prepare_atomic_k2(
            dataclasses.replace(whole, template=template),
            A,
            B,
            stores[template],
            arch,
        )
        launch.launch()
        torch.cuda.synchronize()
    det_out = torch.empty_like(out)
    det_backend.cake_dense_gemm_det_train_router(A, B, det_out, splits=1)
    assert torch.equal(stores[ts], stores[c3]) and torch.equal(stores[ts], det_out)


#: The SIMT programs' tiles ``(tile_n, tile_m, slices)``, compensated ``(tile_n, tile_m, slices, form)``, a test
#: forces (each runs where this checkout registers it).
SIMT_TILES = (
    (64, 64, 2),
    (128, 64, 2),
    (64, 64, 2, "dot2"),
)


def _simt_spec(tile) -> str:
    return "simt_n{}m{}s{}".format(*tile[:3]) + "".join(tile[3:])


def _simt_device_or_skip(tile) -> tuple[torch.device, str, str]:
    device, arch = _atomic_device_or_skip()
    template = cake_backend.router_simt_template(tile)
    if template not in cake_jit.KERNELS[arch]:
        pytest.skip(f"{template} is not registered for {arch}")
    return device, arch, template


@pytest.mark.parametrize("tile", SIMT_TILES, ids=_simt_spec)
@pytest.mark.parametrize(
    ("hidden", "T"),
    [(4096, 1), (4096, 64), (5120, 17), (5120, 192), (6144, 255), (6144, 256)],
)
def test_gpu_router_simt_program_in_its_window(router_rule, tile, hidden, T):
    """A SIMT window (forced for the row on this device) runs the SIMT program of its tile and no other program:
    over ``RERUNS`` launches into a NaN-poisoned ``out`` the weight gradient is finite, within
    ``ROUTER_REL_FRO_MAX`` of FP64 and the same bits every time (every element stored once, its slices added in a
    fixed order), and the prepared launch, the cached launch of ``cake_dense_gemm_nondet_train_router``, ``cake_dense_gemm_nondet_train_router_wgrad`` (its entry
    cache) and a padded ``out`` (written in place, its padding untouched) give those bits."""
    device, arch, template = _simt_device_or_skip(tile)
    sm_count = det_backend.device_facts(device).sm_count
    X, W, G = _router_inputs(T, device, SEED + 40, K=hidden)
    A, B, out = _router_case("wgrad", X, W, G)
    router_rule("wgrad", A, B, _simt_spec(tile))
    plan, reason = cake_backend.plan_atomic_router(
        A, B, out, arch=arch, sm_count=sm_count
    )
    assert plan is not None and plan.simt == tile and plan.template == template, reason
    prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(A, B, out)
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
    assert prepared.plan == plan and not prepared.fused_split
    assert prepared.module_name == cake_jit.select_module(arch, template)
    _assert_ran(prepared.launch, prepared.module_name)
    ref = A.double() @ B.double()
    first = None
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all()), i
        if first is None:
            first = out.clone()
            assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX
        else:
            assert torch.equal(out, first), i
    for _ in range(2):
        cached = torch.full_like(out, float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_router(A, B, cached)
        entry = torch.full_like(out, float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, entry)
        torch.cuda.synchronize()
        assert torch.equal(cached, first) and torch.equal(entry, first)
    _assert_ran(
        lambda: cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, entry),
        prepared.module_name,
    )
    padded = torch.full((256, hidden + 4), float("nan"), device=device)
    cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, padded[:, :hidden])
    torch.cuda.synchronize()
    assert torch.equal(padded[:, :hidden], first)
    assert bool(padded[:, hidden:].isnan().all())


@pytest.mark.parametrize("tile", SIMT_TILES, ids=_simt_spec)
def test_gpu_router_simt_program_replays_in_a_cuda_graph(router_rule, tile):
    """``cake_dense_gemm_nondet_train_router_wgrad`` in a SIMT window captured into a CUDA graph (its entry cache, the prepared launch) replays the
    eager bits into a NaN-poisoned ``out``; the activations and the gradient changed between replays are read
    again."""
    device, arch, template = _simt_device_or_skip(tile)
    hidden, T = 5120, 128
    X, W, G = _router_inputs(T, device, SEED + 41, K=hidden)
    A, B, out = _router_case("wgrad", X, W, G)
    router_rule("wgrad", A, B, _simt_spec(tile))
    eager = torch.empty_like(out)
    cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, eager)
    prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(G.t(), X, out)
    assert prepared.plan.simt == tile
    entry_out = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, entry_out)
            prepared.launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, entry_out)
        prepared.launch()
    for scale in (1.0, -0.5):
        X.mul_(scale)
        want = torch.empty_like(out)
        cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, want)
        entry_out.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(entry_out, want) and torch.equal(out, want), scale
    assert scale != 1.0 and not torch.equal(want, eager)


def test_gpu_router_simt_program_waits_for_an_early_producer(router_rule):
    """The SIMT program behind programmatic dependent launches: the forward writes ``Y``, the input gradient reads it
    and writes ``dX`` (both launched early under PDL), the weight gradient in a SIMT window reads both; launched after
    its predecessor whatever the PDL flag, it reads every value they wrote, with every PDL flag and the same bits."""
    device, arch = _atomic_device_or_skip()
    tile = next(
        (
            t
            for t in SIMT_TILES
            if cake_backend.router_simt_template(t) in cake_jit.KERNELS[arch]
        ),
        None,
    )
    if tile is None:
        pytest.skip(f"no SIMT program is registered for {arch}")
    hidden, T = 6144, 256
    X, W, _ = _router_inputs(T, device, SEED + 81, K=hidden)
    Y = torch.empty(T, 256, device=device)
    dX = torch.empty(T, hidden, device=device)
    dW = torch.empty(256, hidden, device=device)
    router_rule("wgrad", Y.t(), dX, _simt_spec(tile))
    want_y = det_backend.cake_dense_gemm_det_train_router_forward(
        X, W, enable_pdl=False
    )
    want_dx = det_backend.cake_dense_gemm_det_train_router_dgrad(
        want_y, W, enable_pdl=False
    )
    want_dw = torch.empty_like(dW)
    cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
        want_y, want_dx, want_dw, enable_pdl=False
    )
    torch.cuda.synchronize()
    assert (
        _rel_fro(want_dw, want_y.double().t() @ want_dx.double()) <= ROUTER_REL_FRO_MAX
    )
    for pdl in (True, False, None):
        for _ in range(3):
            for t in (Y, dX, dW):
                t.fill_(float("nan"))
            kw = dict(enable_pdl=pdl)
            cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, Y, **kw)
            cake_backend.cake_dense_gemm_nondet_train_router_dgrad(Y, W, dX, **kw)
            cake_backend.cake_dense_gemm_nondet_train_router_wgrad(Y, dX, dW, **kw)
            torch.cuda.synchronize()
            assert torch.equal(Y, want_y) and torch.equal(dX, want_dx), pdl
            assert torch.equal(dW, want_dw), pdl
    assert {key[2][1][1] for key in cake_backend._K2_ATOMIC_CACHE} == {0, 1}


#: The input gradient's SIMT tiles ``(tile_n, tile_m, slices)`` a test forces (each runs where this checkout registers
#: it).
DGRAD_SIMT_TILES = ((64, 32, 4), (64, 32, 8), (64, 64, 4))


def _dgrad_simt_device_or_skip(tile) -> tuple[torch.device, str, str]:
    device, arch = _atomic_device_or_skip()
    template = cake_backend.router_simt_template(tile, "kn")
    if template not in cake_jit.KERNELS[arch]:
        pytest.skip(f"{template} is not registered for {arch}")
    return device, arch, template


@pytest.mark.parametrize("tile", DGRAD_SIMT_TILES, ids=_simt_spec)
@pytest.mark.parametrize(
    ("hidden", "T"),
    [(4096, 1), (4096, 128), (5120, 17), (5120, 64), (6144, 33), (6144, 231)],
)
def test_gpu_router_dgrad_simt_program_in_its_window(router_rule, tile, hidden, T):
    """An input-gradient SIMT window (forced for the row on this device) runs the SIMT program of its tile and no
    other program: over ``RERUNS`` launches into a NaN-poisoned ``out`` the input gradient is finite (every row of a
    partial last row tile written), within ``ROUTER_REL_FRO_MAX`` of FP64 and the same bits every time; the prepared
    launch, the cached launch of ``cake_dense_gemm_nondet_train_router``, ``cake_dense_gemm_nondet_train_router_dgrad`` (its entry cache), operands with padded rows
    (NaN in the padding, read in place) and a padded ``out`` (its padding untouched) give those bits."""
    device, arch, template = _dgrad_simt_device_or_skip(tile)
    sm_count = det_backend.device_facts(device).sm_count
    X, W, G = _router_inputs(T, device, SEED + 140, K=hidden)
    A, B, out = _router_case("dgrad", X, W, G)
    router_rule("dgrad", A, B, _simt_spec(tile))
    plan, reason = cake_backend.plan_atomic_router(
        A, B, out, arch=arch, sm_count=sm_count
    )
    assert plan is not None and plan.simt == tile and plan.template == template, reason
    prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(A, B, out)
    assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
    assert prepared.plan == plan and not prepared.fused_split
    assert prepared.module_name == cake_jit.select_module(arch, template)
    _assert_ran(prepared.launch, prepared.module_name)
    ref = A.double() @ B.double()
    first = None
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all()), i
        if first is None:
            first = out.clone()
            assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX
        else:
            assert torch.equal(out, first), i
    for _ in range(2):
        cached = torch.full_like(out, float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_router(A, B, cached)
        entry = torch.full_like(out, float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, entry)
        torch.cuda.synchronize()
        assert torch.equal(cached, first) and torch.equal(entry, first)
    _assert_ran(
        lambda: cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, entry),
        prepared.module_name,
    )
    G_pad = torch.full((T, 256 + 4), float("nan"), device=device)
    G_pad[:, :256] = G
    W_pad = torch.full((256, hidden + 8), float("nan"), device=device)
    W_pad[:, :hidden] = W
    padded = torch.full((T, hidden + 4), float("nan"), device=device)
    cake_backend.cake_dense_gemm_nondet_train_router_dgrad(
        G_pad[:, :256], W_pad[:, :hidden], padded[:, :hidden]
    )
    torch.cuda.synchronize()
    assert torch.equal(padded[:, :hidden], first)
    assert bool(padded[:, hidden:].isnan().all())


@pytest.mark.parametrize("tile", DGRAD_SIMT_TILES, ids=_simt_spec)
def test_gpu_router_dgrad_simt_program_replays_in_a_cuda_graph(router_rule, tile):
    """``cake_dense_gemm_nondet_train_router_dgrad`` in a SIMT window captured into a CUDA graph (its entry cache, the prepared launch) replays the
    eager bits into a NaN-poisoned ``out``; the output gradient changed between replays is read again."""
    device, arch, template = _dgrad_simt_device_or_skip(tile)
    hidden, T = 5120, 128
    X, W, G = _router_inputs(T, device, SEED + 141, K=hidden)
    A, B, out = _router_case("dgrad", X, W, G)
    router_rule("dgrad", A, B, _simt_spec(tile))
    eager = torch.empty_like(out)
    cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, eager)
    prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(G, W, out)
    assert prepared.plan.simt == tile and prepared.plan.template == template
    entry_out = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, entry_out)
            prepared.launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, entry_out)
        prepared.launch()
    for scale in (1.0, -0.5):
        G.mul_(scale)
        want = torch.empty_like(out)
        cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, want)
        entry_out.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(entry_out, want) and torch.equal(out, want), scale
    assert scale != 1.0 and not torch.equal(want, eager)


def test_gpu_router_dgrad_simt_program_waits_for_an_early_producer(router_rule):
    """The input gradient's SIMT program after programmatic dependent launches: the forward writes ``Y`` (launched
    early under PDL), the input gradient in a SIMT window reads it and writes ``dX``, the weight gradient reads both;
    launched after its predecessor whatever the PDL flag, the SIMT program reads every value the forward wrote, with
    every PDL flag and the same bits."""
    device, arch = _atomic_device_or_skip()
    tile = next(
        (
            t
            for t in DGRAD_SIMT_TILES
            if cake_backend.router_simt_template(t, "kn") in cake_jit.KERNELS[arch]
        ),
        None,
    )
    if tile is None:
        pytest.skip(f"no input-gradient SIMT program is registered for {arch}")
    hidden, T = 6144, 64
    X, W, _ = _router_inputs(T, device, SEED + 142, K=hidden)
    Y = torch.empty(T, 256, device=device)
    dX = torch.empty(T, hidden, device=device)
    dW = torch.empty(256, hidden, device=device)
    router_rule("dgrad", Y, W, _simt_spec(tile))
    # The forward of this row's own route: in a forward SIMT window (64 tokens on
    # 152-SM parts) the cluster program, whose bits differ from the deterministic one.
    want_y = torch.empty_like(Y)
    cake_backend.cake_dense_gemm_nondet_train_router_forward(
        X, W, want_y, enable_pdl=False
    )
    torch.cuda.synchronize()
    assert _rel_fro(want_y, X.double() @ W.double().T) <= ROUTER_REL_FRO_MAX
    want_dx = torch.empty_like(dX)
    cake_backend.cake_dense_gemm_nondet_train_router_dgrad(
        want_y, W, want_dx, enable_pdl=False
    )
    torch.cuda.synchronize()
    assert _rel_fro(want_dx, want_y.double() @ W.double()) <= ROUTER_REL_FRO_MAX
    want_dw = torch.empty_like(dW)
    cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
        want_y, want_dx, want_dw, enable_pdl=False
    )
    torch.cuda.synchronize()
    for pdl in (True, False, None):
        for _ in range(3):
            for t in (Y, dX, dW):
                t.fill_(float("nan"))
            kw = dict(enable_pdl=pdl)
            cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, Y, **kw)
            cake_backend.cake_dense_gemm_nondet_train_router_dgrad(Y, W, dX, **kw)
            cake_backend.cake_dense_gemm_nondet_train_router_wgrad(Y, dX, dW, **kw)
            torch.cuda.synchronize()
            assert torch.equal(Y, want_y) and torch.equal(dX, want_dx), pdl
            assert torch.equal(dW, want_dw), pdl
    simt_keys = [
        key
        for key, e in cake_backend._K2_ATOMIC_CACHE.items()
        if e.plan is not None and e.plan.layout == "kn"
    ]
    assert {key[2][1][1] for key in simt_keys} == {0, 1}


def test_gpu_router_input_gradient_windows_run_at_their_edges():
    """Unforced (no split count): at 1 and 17 tokens and at the first and the last token
    count of every input-gradient SIMT window of this device's router table (``router_rules`` of its architecture and
    SM count), and one token past the last window, ``cake_dense_gemm_nondet_train_router_dgrad`` runs the program the planner gives -- inside a
    window the SIMT program of its tile (the prepared launch's program, within the gate, the same bits on every
    call), outside the deterministic program's bits."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rules, holds = cake_backend.router_rules(arch, sm_count)
    atomic = cake_backend.CakeDenseGemmNondetRouterLaunch
    ran = 0
    for (rule_arch, layout, hidden, experts), rule in sorted(
        rules.items(), key=lambda kv: kv[0]
    ):
        if rule_arch != arch or layout != "kn" or not isinstance(rule, tuple):
            continue
        last = max(hi for _, hi, _ in rule)
        edges = (
            {1, 17, last + 1} | {lo for lo, _, _ in rule} | {hi for _, hi, _ in rule}
        )
        for T in sorted(edges):
            X, W, G = _router_inputs(T, device, SEED + 143 + T, K=hidden)
            spec = next((s for lo, hi, s in rule if lo <= T <= hi), None)
            tile = cake_backend.router_simt(
                "kn", arch=arch, sm_count=sm_count, M=T, N=hidden, K=experts
            )
            assert tile == (cake_backend.router_simt_tile(spec) if holds else None), (
                hidden,
                T,
            )
            prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
                G, W, torch.empty(T, hidden, device=device)
            )
            det_out = det_backend.cake_dense_gemm_det_train_router_dgrad(G, W)
            first = None
            for _ in range(2):
                out = torch.full((T, hidden), float("nan"), device=device)
                assert (
                    cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out)
                    is out
                )
                torch.cuda.synchronize()
                if tile is None:
                    assert torch.equal(out, det_out), (hidden, T)
                    continue
                assert _rel_fro(out, G.double() @ W.double()) <= ROUTER_REL_FRO_MAX
                if first is None:
                    first = out.clone()
                assert torch.equal(out, first), (hidden, T)
            if tile is None:
                assert not isinstance(prepared, atomic), (hidden, T)
                continue
            assert isinstance(prepared, atomic), (hidden, T)
            assert tuple(prepared.plan.simt) == tile, (hidden, T)
            _assert_ran(
                lambda G=G,
                W=W,
                out=out: cake_backend.cake_dense_gemm_nondet_train_router_dgrad(
                    G, W, out
                ),
                prepared.module_name,
            )
            ran += 1
    print(
        f"ROUTER_DGRAD_WINDOW_EDGES {arch} sm_count={sm_count}: {ran} token counts ran"
    )
    if not ran:
        pytest.skip(
            f"no input-gradient SIMT window of {arch} applies on {sm_count} SMs"
        )


def _device_simt_windows(arch, sm_count, layout) -> list:
    """The shipped SIMT windows of the 148-SM tables of this test (:data:`ROUTER_WGRAD_WINDOWS_BY_SM`,
    :data:`ROUTER_DGRAD_WINDOWS`; plain and compensated) that hold on a part of ``arch`` with ``sm_count`` SMs,
    ``[(hidden, first, last, spec), ...]`` for the layout class ``layout`` (256 experts)."""
    if sm_count != 148:
        return []
    rows = []
    for hidden in (4096, 5120, 6144):
        if layout == "kn":
            windows = ROUTER_DGRAD_WINDOWS.get((arch, sm_count, hidden), ())
        else:
            windows = ROUTER_WGRAD_WINDOWS_BY_SM.get((arch, sm_count, hidden), ())
        rows += [
            (hidden, first, last, spec)
            for first, last, spec in windows
            if cake_backend.router_simt_tile(spec) is not None
        ]
    return rows


def test_gpu_router_forward_simt_windows_keep_the_fp64_rule(monkeypatch):
    """Unforced (``cake_dense_gemm_nondet_train_router_forward(...)``, no split count), at the first and the last token count of
    each forward SIMT window of the device's part (on a part with a router table of its own its windows,
    :data:`ROUTER_FWD_WINDOWS_BY_SM`; else those of the architecture, :data:`ROUTER_FWD_WINDOWS`, and on a part whose
    SM count is not :data:`ROUTER_SM_COUNT` the measured count is set to the device's, so the program runs there too)
    and two seeds: the planner gives the window's cluster program, which the call launches (no program of the
    deterministic package); three calls into a NaN-poisoned ``out`` give finite values and the same bits (every element
    stored once, the ranks' partials added in rank order); the relative Frobenius error against FP64 of the exact
    operands is within ``ROUTER_REL_FRO_MAX``, every element within the error bound of a sum of ``K`` FP32 products in
    any order, and the top-8 experts of every row are FP64's except near ties; a CUDA graph captured after the eager
    call replays the same bits.  The ratio to the error of the deterministic program is printed, not checked.  A part
    with a router table of its own and no forward window in it runs the deterministic program at the token counts of
    the architecture's windows."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    arch_windows = [
        (h, a, b, s)
        for (ar, h), w in ROUTER_FWD_WINDOWS.items()
        if ar == arch
        for a, b, s in w
    ]
    own = (arch, sm_count) in cake_backend.ROUTER_RULES_BY_SM
    assert own == ((arch, sm_count) in ROUTER_BY_SM_PARTS)
    windows = (
        [
            (h, a, b, s)
            for (ar, sms, h), w in ROUTER_FWD_WINDOWS_BY_SM.items()
            if (ar, sms) == (arch, sm_count)
            for a, b, s in w
        ]
        if own
        else arch_windows
    )
    if not windows:
        for hidden, first, last, _ in arch_windows:
            for T in (first, last):
                kw = dict(arch=arch, M=T, N=256, K=hidden, sm_count=sm_count)
                assert cake_backend.router_forward_simt(**kw) is None, (hidden, T)
        print(
            f"ROUTER_FWD_SIMT_FP64 {arch} sm_count={sm_count}: no forward window on this part"
        )
        return
    if not own and sm_count != cake_backend.ROUTER_SM_COUNT[arch]:
        monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
    ran = 0
    for hidden, first, last, spec in windows:
        tile = cake_backend.router_forward_simt_tile(spec)
        template = cake_backend.router_forward_simt_template(tile)
        _require_template(arch, template)
        module = cake_jit.select_module(arch, template)
        for T in sorted({first, last}):
            for seed in (SEED + 250, SEED + 251):
                X, W, _ = _router_inputs(T, device, seed, K=hidden)
                A, B, out = _router_case("fwd", X, W, None)
                plan, reason = cake_backend.plan_atomic_router(
                    A, B, out, arch=arch, sm_count=sm_count
                )
                assert plan is not None, (hidden, T, reason)
                assert plan.simt == tile and plan.template == template, (hidden, T)
                cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                got = None
                for i in range(3):
                    out.fill_(float("nan"))
                    cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, out)
                    torch.cuda.synchronize()
                    assert bool(torch.isfinite(out).all()), (hidden, T, i)
                    if got is None:
                        got = out.clone()
                    else:
                        assert torch.equal(out, got), (hidden, T, i)
                if seed == SEED + 250:
                    _assert_ran(
                        lambda: cake_backend.cake_dense_gemm_nondet_train_router_forward(
                            X, W, out
                        ),
                        module,
                    )
                    graph = torch.cuda.CUDAGraph()
                    out.fill_(float("nan"))
                    with torch.cuda.graph(graph):
                        cake_backend.cake_dense_gemm_nondet_train_router_forward(
                            X, W, out
                        )
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, got), (hidden, T)
                    del graph
                ref = X.double() @ W.double().t()
                base = det_backend.cake_dense_gemm_det_train_router_forward(X, W)
                err, base_err = _rel_fro(got, ref), _rel_fro(base, ref)
                assert err <= ROUTER_REL_FRO_MAX, (hidden, T, seed, err)
                gamma = hidden * 2.0**-24 / (1 - hidden * 2.0**-24)
                bound = gamma * (X.double().abs() @ W.double().abs().t())
                assert bool(((got.double() - ref).abs() <= bound).all()), (hidden, T)
                differ, ties = _top8_rows_differ(got, ref)
                assert differ == 0, (hidden, T, differ, ties)
                print(
                    f"ROUTER_FWD_SIMT_FP64 {arch} sm_count={sm_count} hidden={hidden} T={T} seed={seed} {spec}: "
                    f"rel_fro {err:.4e}, deterministic program {base_err:.4e}, ratio {err / base_err:.4f}"
                )
                ran += 1
        # one token past the window's last: another spec (the next window's, or the deterministic program)
        past = cake_backend._router_rule_spec(
            "kk", arch=arch, sm_count=sm_count, M=last + 1, N=256, K=hidden
        )
        assert past != spec, (hidden, last + 1, past)
    assert ran == sum(2 * len({first, last}) for _, first, last, _ in windows)


@pytest.mark.parametrize("op", ["wgrad", "dgrad"])
def test_gpu_router_shipped_simt_windows_keep_the_fp64_rule(monkeypatch, op):
    """Unforced (no split count), at the first and the last token count of each shipped
    SIMT window of the 148-SM tables that holds on this device (148-SM sm_100a / sm_103a parts;
    :func:`_device_simt_windows`) and two seeds: the planner gives the window's program; three calls into a
    NaN-poisoned ``out`` give finite values and the same bits; the relative Frobenius error against FP64 of the exact
    operands is within ``ROUTER_REL_FRO_MAX`` (the accuracy rule of the windows) and every element within the error
    bound of a sum of ``K`` FP32 products in any order (``gamma_K`` times the sum of the products' magnitudes).  The
    ratio to the error of the plan the row takes without its SIMT windows (the deterministic program) is printed, not
    checked.  One token past a window takes another plan.  On every other part no token count of these windows runs
    a compensated SIMT program (a 152-SM or 212-SM part may run the plain SIMT program of its own window there)."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    layout = "nn_t" if op == "wgrad" else "kn"
    windows = _device_simt_windows(arch, sm_count, layout)
    if not windows:
        tokens = {
            (hidden, T)
            for part_arch in cake_jit.ARCHES
            for sms in ROUTER_SM_FACTS
            for hidden, first, last, _ in _device_simt_windows(part_arch, sms, layout)
            for T in (first, last)
        }
        assert tokens
        for hidden, T in sorted(tokens):
            M, N, K = _router_launch_dims(layout, hidden, 256, T)
            kw = dict(arch=arch, sm_count=sm_count, M=M, N=N, K=K)
            simt = cake_backend.router_simt(layout, **kw)
            assert simt is None or len(simt) == 3, (hidden, T, simt)
        print(f"ROUTER_SIMT_FP64 {arch} sm_count={sm_count} {op}: no shipped window")
        return
    entry = (
        cake_backend.cake_dense_gemm_nondet_train_router_wgrad
        if op == "wgrad"
        else cake_backend.cake_dense_gemm_nondet_train_router_dgrad
    )
    rules, holds = cake_backend.router_rules(arch, sm_count)
    assert holds
    ran = 0
    for hidden, first, last, spec in windows:
        tile = cake_backend.router_simt_tile(spec)
        assert tile is not None, spec
        template = cake_backend.router_simt_template(tile, layout)
        _require_template(arch, template)
        key = (arch, layout, hidden, 256)
        without = (
            tuple(w for w in rules[key] if cake_backend.router_simt_tile(w[2]) is None)
            or None
        )
        for T in sorted({first, last}):
            for seed in (SEED + 240, SEED + 241):
                X, W, G = _router_inputs(T, device, seed, K=hidden)
                A, B, out = _router_case(op, X, W, G)
                args = (G, X) if op == "wgrad" else (G, W)
                plan, reason = cake_backend.plan_atomic_router(
                    A, B, out, arch=arch, sm_count=sm_count
                )
                assert plan is not None, (hidden, T, reason)
                assert plan.simt == tile and plan.template == template, (hidden, T)
                cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                got = None
                for i in range(3):
                    out.fill_(float("nan"))
                    entry(*args, out)
                    torch.cuda.synchronize()
                    assert bool(torch.isfinite(out).all()), (hidden, T, i)
                    if got is None:
                        got = out.clone()
                    else:
                        assert torch.equal(out, got), (hidden, T, i)
                with monkeypatch.context() as m:
                    m.setitem(rules, key, without)
                    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                    base = torch.full_like(out, float("nan"))
                    entry(*args, base)
                    torch.cuda.synchronize()
                cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                ref = A.double() @ B.double()
                err, base_err = _rel_fro(got, ref), _rel_fro(base, ref)
                print(
                    f"ROUTER_SIMT_FP64 {arch} sm_count={sm_count} {op} hidden={hidden} T={T} seed={seed}: "
                    f"rel_fro {err:.4e}, without the window {base_err:.4e}, ratio {err / base_err:.4f}"
                )
                assert err <= ROUTER_REL_FRO_MAX, (hidden, T, err)
                k = A.shape[-1]
                gamma = k * 2.0**-24 / (1 - k * 2.0**-24)
                bound = gamma * (A.double().abs() @ B.double().abs())
                assert bool(((got.double() - ref).abs() <= bound).all()), (hidden, T)
                ran += 1
        # one token past the window: another plan
        M, N, K = _router_launch_dims(layout, hidden, 256, last + 1)
        past = cake_backend._router_rule_spec(
            layout, arch=arch, sm_count=sm_count, M=M, N=N, K=K
        )
        assert past != spec, (hidden, last + 1, past)
    assert ran == sum(2 * len({first, last}) for _, first, last, _ in windows)


def test_gpu_router_entry_points_reach_the_stream_k_program(router_rule):
    """``cake_dense_gemm_nondet_train_router_forward`` / ``cake_dense_gemm_nondet_train_router_wgrad`` with a verdict for each row (no forward
    verdict ships; the weight gradient of this hidden size runs the deterministic plan): the forward repeats bit
    for bit, and its top-8, like the deterministic forward's, is FP64's on every row but the near ties (the 8th and
    9th logits within 16 FP32 ulps); both meet the gate."""
    device, arch = _atomic_device_or_skip()
    X, W, G = _router_inputs(ROUTER_T, device, SEED + 21)
    router_rule("fwd", X, W.t(), "forward")
    router_rule("wgrad", G.t(), X, "pairs")
    first = cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W)
    det_fwd = det_backend.cake_dense_gemm_det_train_router_forward(X, W)
    ref_fwd, ref_wgrad = X.double() @ W.double().t(), G.double().t() @ X.double()
    for name, out in (("atomic", first), ("det", det_fwd)):
        differ, ties = _top8_rows_differ(out, ref_fwd)
        print(
            f"ROUTER_TOP8 {arch} {name}: {differ} rows differ, {ties} near ties excluded"
        )
        assert differ == 0 and ties <= ROUTER_T // 100, (differ, ties)
    for _ in range(RERUNS):
        out = cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W)
        assert torch.equal(out, first) and _rel_fro(out, ref_fwd) <= ROUTER_REL_FRO_MAX
        wgrad = cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X)
        assert _rel_fro(wgrad, ref_wgrad) <= ROUTER_REL_FRO_MAX
    plan = cake_backend.cake_dense_gemm_nondet_train_router_plan(X, W.t(), first)
    assert isinstance(plan, cake_backend.CakeDenseGemmNondetRouterPlan)
    _assert_ran(
        lambda: cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, first),
        cake_jit.select_module(arch, plan.template),
    )


def test_gpu_planned_rows_run_the_programs_of_their_verdicts():
    """Unforced (no split count): every K1 weight-gradient row with a verdict for this
    architecture and every measured router row prepares the program its verdict names on this device (by kernel
    name, at ``T = 16231``; a row with a range of :data:`K1_DSM_RANGES_BY_SM` there the range's cluster program, else
    one with a range of :data:`K1_ORDERED_RANGES_BY_SM` there that range's program), and
    a row whose verdict is ``None`` -- or that this device's SM count leaves to the deterministic program -- prepares
    the deterministic launch."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    g = torch.Generator(device=device).manual_seed(SEED + 90)
    T = 16231
    ran = 0
    keys = (
        set(cake_backend.K1_SPLIT_RULES)
        | set(cake_backend.K1_SPLIT_RULES_BY_SM.get((arch, sm_count), {}))
        | set(cake_backend.K1_SPLIT_OVERRIDES_BY_SM.get((arch, sm_count), {}))
    )
    for rule_arch, L, M, N, dt in sorted(keys):
        if rule_arch != arch:
            continue
        G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
        out_dtype = torch.float32 if dt == "f32" else torch.bfloat16
        out = torch.empty(N, M, device=device, dtype=out_dtype)
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
            G, X, out
        )
        kw = dict(arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sm_count)
        dsm = cake_backend.k1_dsm_plan(**kw)
        odd = cake_backend.k1_dsm_odd_block_plan(**kw) if dsm else None
        if odd is not None:
            # an odd number of K blocks in a range of K1_DSM_ODD_BLOCK_RANGES_BY_SM runs that table's program
            dsm, ordered = None, odd or None
            want = tuple(odd[:2]) if odd else None
        else:
            # a token count of a K1_ORDERED_RANGES_BY_SM range runs that range's program in place of the verdict
            ordered = None if dsm else cake_backend.k1_ordered_plan(**kw)
            want = dsm or (
                tuple(ordered[:2]) if ordered else cake_backend.k1_plan(**kw)
            )
        if want is None:
            assert not isinstance(
                prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
            ), (M, N, dt)
            continue
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch), (
            M,
            N,
            dt,
        )
        assert (prepared.plan.block_n, prepared.plan.splits) == want
        assert prepared.plan.dsm == (dsm is not None), (M, N, dt)
        if ordered:
            assert (prepared.plan.chains, prepared.plan.split_order) == tuple(
                ordered[2:]
            ), (M, N, dt)
        _assert_ran(prepared.launch, prepared.module_name)
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double(), f"{M} {N} {dt}")
        ran += 1
    rules, _ = cake_backend.router_rules(arch, sm_count)
    for (rule_arch, layout, hidden, _experts), spec in sorted(
        rules.items(), key=lambda kv: kv[0]
    ):
        if rule_arch != arch:
            continue
        X, W, G = _router_inputs(T, device, SEED + 91, K=hidden)
        op = {"kk": "fwd", "kn": "dgrad", "nn_t": "wgrad"}[layout]
        A, B, out = _router_case(op, X, W, G)
        prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(A, B, out)
        plan, _ = cake_backend.plan_atomic_router(
            A, B, out, arch=arch, sm_count=sm_count
        )
        if plan is None:
            assert not isinstance(
                prepared, cake_backend.CakeDenseGemmNondetRouterLaunch
            )
            continue
        assert spec is not None
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
        assert (prepared.plan.template, prepared.plan.units) == (
            plan.template,
            plan.units,
        )
        assert (prepared.plan.head_spans, prepared.plan.head_steps) == (
            plan.head_spans,
            plan.head_steps,
        )
        _assert_ran(prepared.launch, prepared.module_name)
        torch.cuda.synchronize()
        assert _rel_fro(out, A.double() @ B.double()) <= ROUTER_REL_FRO_MAX
        ran += 1
    print(
        f"PLANNED_ROWS {arch} sm_count={sm_count}: {ran} rows ran this package's programs"
    )
    if not ran:
        pytest.skip(f"no verdict of {arch} applies on {sm_count} SMs")


def test_gpu_k1_sm_count_overrides_run_their_programs():
    """Unforced (no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_OVERRIDES_BY_SM` (``sm_107a``: 216 SMs): every overridden row prepares and runs the override's
    program at the first and the last token count of :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and one token below the last,
    outside the part's ranges of :data:`K1_CHAIN_RANGES_BY_SM`, which come first, and its token counts of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` (``4096 -> 1024`` FP32 at 16383 tokens: the 192-wide row-major-store program
    at one split), meets the deterministic package's FP64 budget and, where both run one split, gives the bits
    of the row's :data:`K1_SPLIT_RULES` program (256-wide transposed tiles) on the same inputs.  A part of any other SM
    count skips."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    table = cake_backend.K1_SPLIT_OVERRIDES_BY_SM.get((arch, sm_count), {})
    rows = sorted((key, choice) for key, choice in table.items() if choice is not None)
    if not rows:
        pytest.skip(f"no K1 override of {arch} on {sm_count} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 93)
    ran = 0
    for key, (block_n, splits) in rows:
        _, L, M, N, dt = key
        f32 = dt == "f32"
        rule = cake_backend.K1_SPLIT_RULES[key]
        lo_t, hi_t = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
        for T in (lo_t, hi_t - 1, hi_t):
            kw = dict(arch=arch, L=L, M=M, N=N, K=T, out_f32=f32, sm_count=sm_count)
            # the ranges of K1_CHAIN_RANGES_BY_SM come first
            if cake_backend.k1_chain_plan(**kw) is not None:
                continue
            want = cake_backend.k1_plan(**kw)
            if want is None:  # a token count of K1_SPLIT_RULE_EXCLUDED_TOKENS
                continue
            assert want == (block_n, splits), (key, T)
            G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
            out_dtype = torch.float32 if f32 else torch.bfloat16
            out = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
            prepared = (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                    G, X, out
                )
            )
            assert isinstance(
                prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
            ), (key, T)
            assert (prepared.plan.block_n, prepared.plan.splits) == want
            assert prepared.template == cake_backend.atomic_k1_template(block_n, f32)
            _assert_ran(prepared.launch, prepared.module_name)
            torch.cuda.synchronize()
            _k1_check(out, G.double().t() @ X.double(), f"{M} {N} {dt} {T}")
            if rule is not None and rule[1] == 1 and splits == 1:
                same = torch.full_like(out, float("nan"))
                recorded = _k1_prepare(G, X, same, arch, 1, rule[0])
                assert recorded.template != prepared.template
                _assert_ran(recorded.launch, recorded.module_name)
                torch.cuda.synchronize()
                assert torch.equal(out, same), (key, T)
            ran += 1
    print(
        f"K1_OVERRIDES {arch} sm_count={sm_count}: {ran} launches ran the override programs"
    )
    assert ran


def test_gpu_k1_forward_rows_run_their_plan():
    """On a part of exactly the SM count of a table of :data:`K1_FORWARD_RULES_BY_SM` (152-SM ``sm_100a`` /
    ``sm_103a``): the forward of every row binds the program the deterministic planner
    gives the row with the table's knobs (the single-CTA program), meets the deterministic package's BF16 budget
    against FP64, repeats bit for bit through the bound launch, and the deterministic package's forward meets the
    budget too.  A
    part of any other SM count skips."""
    device = _det_device_or_skip()
    facts = det_backend.device_facts(device)
    rows = {
        key: knobs
        for key, knobs in cake_backend.K1_FORWARD_RULES_BY_SM.get(
            (facts.arch, facts.sm_count), {}
        ).items()
        if not isinstance(knobs, str)
    }
    if not rows:
        pytest.skip(f"no K1 forward knob row of {facts.arch} on {facts.sm_count} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 160)
    ran = 0
    for (K, N, T), knobs in sorted(rows.items()):
        X, W = _bf16((T, K), device, g), _bf16((N, K), device, g)
        ref = X.double() @ W.double().t()
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full((T, N), float("nan"), device=device, dtype=torch.bfloat16)
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, out)
            is out
        )
        torch.cuda.synchronize()
        ((_, entry),) = cake_backend._K1_DET_CACHE.items()
        want, *_ = det_backend._k1_plan(
            X, W.t(), out, transposed_out=False, knobs=dict(knobs)
        )
        assert entry.plan == want and entry.k1 is not None, (K, N, T)
        assert entry.plan.template.endswith("_kk_n128_c1"), (K, N, T)
        _k1_check(out, ref, f"forward {K} {N} {T}")
        again = torch.full_like(out, float("nan"))
        cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, again)
        torch.cuda.synchronize()
        assert torch.equal(out, again), (K, N, T)
        det_out = torch.full_like(out, float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_forward(X, W, det_out)
        torch.cuda.synchronize()
        _k1_check(det_out, ref, f"deterministic forward {K} {N} {T}")
        ran += 1
    print(
        f"K1_FORWARD_ROWS {facts.arch} sm_count={facts.sm_count}: {ran} rows ran their plan"
    )
    assert ran


def test_gpu_k1_sm_count_overrides_through_the_convenience_binding():
    """Unforced (no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_OVERRIDES_BY_SM` (``sm_107a``: 216 SMs): the convenience path of every overridden row binds the
    override's program (its :data:`_K1_ATOMIC_CACHE` entry) at the first and the last token count of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and one token below the last, outside the part's ranges of
    :data:`K1_CHAIN_RANGES_BY_SM` and its token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`, and gives the bits of
    the prepared launch, the tickets zero after.  A part of any other SM count skips."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    table = cake_backend.K1_SPLIT_OVERRIDES_BY_SM.get((arch, sm_count), {})
    rows = sorted((key, choice) for key, choice in table.items() if choice is not None)
    if not rows:
        pytest.skip(f"no K1 override of {arch} on {sm_count} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 94)
    ran = 0
    for key, (block_n, splits) in rows:
        _, L, M, N, dt = key
        f32 = dt == "f32"
        lo_t, hi_t = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
        for T in (lo_t, hi_t - 1, hi_t):
            kw = dict(arch=arch, L=L, M=M, N=N, K=T, out_f32=f32, sm_count=sm_count)
            # the ranges of K1_CHAIN_RANGES_BY_SM come first
            if cake_backend.k1_chain_plan(**kw) is not None:
                continue
            plan = cake_backend.k1_plan(**kw)
            if plan != (block_n, splits):  # an excluded token count
                continue
            G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
            out_dtype = torch.float32 if f32 else torch.bfloat16
            want = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
            prepared = (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                    G, X, want
                )
            )
            prepared.launch()
            cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
            out = torch.full_like(want, float("nan"))
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                is out
            )
            torch.cuda.synchronize()
            (entry,) = [
                e for e in cake_backend._K1_ATOMIC_CACHE.values() if e.plan is not None
            ]
            assert entry.k1 is not None, (key, T)
            assert (entry.plan.block_n, entry.plan.splits) == plan, (key, T)
            if splits <= 2:  # at most two partials per element: bitwise reproducible
                assert torch.equal(out, want), (key, T)
            else:
                _k1_check(out, G.double().t() @ X.double(), f"{M} {N} {dt} {T}")
            assert _arena_tickets_zero()
            ran += 1
    assert ran


def _k1_token_range_edges(arch, sm_count):
    """``(key, choice, T, want)`` for every row of the part's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`,
    :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` and :data:`K1_TWO_SPLIT_RANGES_BY_SM`: the first and the last token count of
    each range (``want``: the program of the part's table, or of the override or two-split range), one token past the
    first token count of a two-split range (an odd number of 64-token K blocks; ``want``: that range's program), and one
    token outside a range where neither the window nor another range of the row holds (``want`` ``None``: the
    deterministic program); ``choice`` is the planner's."""
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    part = (arch, sm_count)
    rows = cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES.get(part, {})
    over = cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM.get(part, {})
    two = cake_backend.K1_TWO_SPLIT_RANGES_BY_SM.get(part, {})
    table = cake_backend.K1_SPLIT_RULES_BY_SM.get(part, {})
    cases = []
    for key in sorted(set(rows) | set(over) | set(two)):
        _, L, M, N, dt = key
        spans = [(lo, hi, table[key], False) for lo, hi in rows.get(key, ())]
        spans += [(lo, hi, v, False) for lo, hi, v in over.get(key, ())]
        spans += [(lo, hi, v, True) for lo, hi, v in two.get(key, ())]
        for lo, hi, _, split in spans:
            for T in sorted({lo, hi, lo - 1, hi + 1} | ({lo + 1} if split else set())):
                held = [v for a, b, v, _ in spans if a <= T <= b]
                if not held and first <= T <= last:
                    continue
                plan = cake_backend.k1_plan(
                    arch=arch,
                    L=L,
                    M=M,
                    N=N,
                    K=T,
                    out_f32=dt == "f32",
                    sm_count=sm_count,
                )
                cases.append((key, plan, T, tuple(held[0]) if held else None))
    return cases


def _k1_exact_sm_count_edges(arch, sm_count):
    """``(key, choice, T, want)`` for every row of the part's table in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`
    (exactly ``sm_count``): the first and the last token count of each range (65537, 67520 and 67584 -- an odd and an
    even number of 64-token blocks -- for a range without an upper bound; ``want`` ``None``: the deterministic program)
    and one token outside it (``want``: the row's :data:`K1_SPLIT_RULES` verdict, or ``None`` where an entry of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` or :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` for the part holds the token
    count); ``choice`` is the planner's."""
    rows = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM.get((arch, sm_count), {})
    cases = []
    for key, spans in sorted(rows.items()):
        _, L, M, N, dt = key
        rule = cake_backend.K1_SPLIT_RULES[key]
        floor = max(
            cake_backend.K1_MIN_SM_COUNT[arch],
            cake_backend.K1_RULE_MIN_SM_COUNT.get(key, 0),
        )

        def held(T, key=key, spans=spans, floor=floor):
            general = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS.get(key, ())
            if _k1_blocks(T) % 2:
                general += cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS.get(key, ())
            # an entry at or above the row's SM-count floor holds from its SM count up, one below it on its SM count
            on_part = [
                (lo, hi)
                for sms, lo, hi in general
                if (sm_count >= sms if sms >= floor else sm_count == sms)
            ]
            return any(
                lo <= T and (hi is None or hi >= T) for lo, hi in (*spans, *on_part)
            )

        for lo, hi in spans:
            ends = {lo} | ({65537, 67520, 67584} if hi is None else {hi})
            for T in sorted(ends | {lo - 1} | (set() if hi is None else {hi + 1})):
                plan = cake_backend.k1_plan(
                    arch=arch,
                    L=L,
                    M=M,
                    N=N,
                    K=T,
                    out_f32=dt == "f32",
                    sm_count=sm_count,
                )
                cases.append((key, plan, T, None if held(T) else tuple(rule)))
    return cases


def _k1_one_split_range(arch, sm_count, key, T):
    """The tile width of the range of :data:`K1_ONE_SPLIT_RANGES_BY_SM` that holds token count ``T`` of row ``key`` on
    a part of ``sm_count`` SMs, or ``None``."""
    ranges = cake_backend.K1_ONE_SPLIT_RANGES_BY_SM.get((arch, sm_count), {})
    held = [bn for lo, hi, bn in ranges.get(key, ()) if lo <= T <= hi]
    return held[0] if held else None


def _k1_with_chain_ranges(arch, sm_count, key, T, want):
    """``(want, chains)``: an edge case's expected ``(block_n, splits)`` and chain count with
    :data:`K1_CHAIN_RANGES_BY_SM` and then :data:`K1_ONE_SPLIT_RANGES_BY_SM`, whose ranges come before the tables the
    case is drawn from (:func:`k1_plan`): a token count of a chain range expects that range's tile width at one split
    and its chain count, a token count of a one-split range that range's tile width at one split and one chain, every
    other the case's ``want`` (one chain)."""
    _, L, M, N, dt = key
    chain = cake_backend.k1_chain_plan(
        arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sm_count
    )
    if chain is not None:
        return (chain[0], 1), chain[1]
    one = _k1_one_split_range(arch, sm_count, key, T)
    return (want, 1) if one is None else ((one, 1), 1)


#: Edge token counts of the GPU edge test at which the deterministic program itself misses the BF16 budget for some
#: seed (measured, 10 seeds), so the budget does not choose the route there: ``(arch, sm_count, key, T) -> T'``.  The
#: edge asserts the route only, and the accuracy rule runs at ``T'``, the nearest measured token count of the same
#: range at which the deterministic program met the budget on every seed on the measured parts (``None``: none).
K1_P1_EDGES = {
    ("sm_100a", 148, ("sm_100a", 1, 4096, 64, "bf16"), 65535): 41985,
    ("sm_103a", 148, ("sm_103a", 1, 4096, 32, "bf16"), 65535): 45057,
    ("sm_103a", 148, ("sm_103a", 1, 5120, 48, "bf16"), 49152): None,
    ("sm_103a", 148, ("sm_103a", 1, 5120, 64, "bf16"), 47104): 45057,
    ("sm_103a", 148, ("sm_103a", 1, 5120, 64, "bf16"), 49153): None,
    ("sm_103a", 148, ("sm_103a", 1, 5120, 128, "bf16"), 65537): None,
}


def test_gpu_k1_sm_count_token_ranges_run_at_their_edges(no_k1_dsm, monkeypatch):
    """Unforced (no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` or
    :data:`K1_TWO_SPLIT_RANGES_BY_SM` (``sm_107a``: 212 SMs; ``sm_100a`` / ``sm_103a``: 152 SMs): at the first and the
    last token count of every range of a row, and one token past the first token count of a two-split range (an odd
    number of K blocks), the prepared launch runs the program of the part's table in :data:`K1_SPLIT_RULES_BY_SM` (of
    the range, for an override or two-split range) and meets the K1 accuracy rule (:func:`_k1_check`); one token outside
    a range (and outside the window) the row runs the deterministic program, with the deterministic package's bits.  On
    a part of exactly the SM count of a table of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` (148-SM ``sm_100a`` /
    ``sm_103a`` parts) the reverse: the deterministic program with its bits at both ends of a range, the row's program
    one token outside it.  An edge inside a range of :data:`K1_CHAIN_RANGES_BY_SM` for the part runs that range's
    program of several chains instead (:func:`_k1_with_chain_ranges`), under the same FP64 rule, and an edge inside a
    range of :data:`K1_ONE_SPLIT_RANGES_BY_SM` that range's one-split program, under the same rule.  At an edge of
    ``K1_P1_EDGES`` the deterministic program itself misses the BF16 budget for some seed: the edge asserts the route
    only, and the FP64 rule runs at the entry's token count of the same range instead (none: no check).  The cluster
    ranges of :data:`K1_DSM_RANGES_BY_SM`, which come before these tables, are taken out
    (``test_gpu_k1_dsm_ranges_run_at_their_edges`` holds them), and so are the ranges of
    :data:`K1_PARITY_RANGES_BY_SM` (``test_k1_parity_ranges_bind_per_token_count`` holds them).  A part of any other SM
    count skips."""
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    cases = _k1_token_range_edges(arch, sm_count)
    cases += _k1_exact_sm_count_edges(arch, sm_count)
    if not cases:
        pytest.skip(f"no K1 token range of {arch} on {sm_count} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 95)
    ran = 0
    for key, plan, T, want_plan in cases:
        _, L, M, N, dt = key
        f32 = dt == "f32"
        want_plan, chains = _k1_with_chain_ranges(arch, sm_count, key, T, want_plan)
        G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
        out_dtype = torch.float32 if f32 else torch.bfloat16
        out = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
            G, X, out
        )
        if want_plan is not None:
            assert plan == want_plan, (key, T)
            assert isinstance(
                prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
            ), (key, T)
            assert (prepared.plan.block_n, prepared.plan.splits) == plan, (key, T)
            assert prepared.plan.chains == chains, (key, T)
            assert prepared.template == cake_backend.atomic_k1_template(
                plan[0], f32, chains
            ), (key, T)
            _assert_ran(prepared.launch, prepared.module_name)
            torch.cuda.synchronize()
            ran += 1
            at = K1_P1_EDGES.get((arch, sm_count, key, T), T)
            if at is None:
                continue
            if at != T:
                G, X = _bf16((at, N), device, g), _bf16((at, M), device, g)
                out = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
                prepared = (
                    cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                        G, X, out
                    )
                )
                assert isinstance(
                    prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
                ), (key, at)
                assert (prepared.plan.block_n, prepared.plan.splits) == plan, (key, at)
                _assert_ran(prepared.launch, prepared.module_name)
                torch.cuda.synchronize()
            _k1_check(out, G.double().t() @ X.double(), f"{M} {N} {dt} {at}")
        else:
            assert plan is None, (key, T)
            assert not isinstance(
                prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
            ), (key, T)
            prepared.launch()
            want = torch.full_like(out, float("nan"))
            det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X, want)
            torch.cuda.synchronize()
            assert torch.equal(out, want), (key, T)
    print(
        f"K1_TOKEN_RANGES {arch} sm_count={sm_count}: {ran} launches ran this package's programs at range edges"
    )
    assert ran


def test_gpu_k1_sm_count_token_ranges_through_the_convenience_binding(
    no_k1_dsm, monkeypatch
):
    """Unforced (no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` or
    :data:`K1_TWO_SPLIT_RANGES_BY_SM` (``sm_107a``: 212 SMs; ``sm_100a`` / ``sm_103a``: 152 SMs): the
    convenience path binds the part's program
    (the range's, for an override or two-split range) at the first and the last token count of every range of a row
    (its :data:`_K1_ATOMIC_CACHE` entry) and gives the bits of the prepared launch where at most two partials meet (the
    FP64 budget otherwise), the tickets zero after; one token outside a range, and one token past the first token count
    of a two-split range, it binds the deterministic package's plan.  On a part of exactly the SM count of a table
    of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` the reverse.  An edge inside a range of :data:`K1_CHAIN_RANGES_BY_SM`
    for the part binds that range's program of several chains instead (one split: the bits of the prepared launch).  The
    cluster ranges of :data:`K1_DSM_RANGES_BY_SM` and the ranges of :data:`K1_PARITY_RANGES_BY_SM` are taken out, as
    in ``test_gpu_k1_sm_count_token_ranges_run_at_their_edges``.  A part of any other SM count skips."""
    monkeypatch.setattr(cake_backend, "K1_PARITY_RANGES_BY_SM", {})
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    cases = _k1_token_range_edges(arch, sm_count)
    cases += _k1_exact_sm_count_edges(arch, sm_count)
    if not cases:
        pytest.skip(f"no K1 token range of {arch} on {sm_count} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 96)
    ran = 0
    for key, plan, T, want_plan in cases:
        _, L, M, N, dt = key
        f32 = dt == "f32"
        want_plan, chains = _k1_with_chain_ranges(arch, sm_count, key, T, want_plan)
        G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
        out_dtype = torch.float32 if f32 else torch.bfloat16
        want = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
            G, X, want
        )
        prepared.launch()
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full_like(want, float("nan"))
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out) is out
        )
        torch.cuda.synchronize()
        (entry,) = list(cake_backend._K1_ATOMIC_CACHE.values())
        assert entry.k1 is not None, (key, T)
        if want_plan is not None:
            assert plan == want_plan, (key, T)
            assert (entry.plan.block_n, entry.plan.splits) == plan, (key, T)
            assert entry.plan.chains == chains, (key, T)
            if plan[1] <= 2:  # at most two partials per element: bitwise reproducible
                assert torch.equal(out, want), (key, T)
            else:
                _k1_check(out, G.double().t() @ X.double(), f"{M} {N} {dt} {T}")
            assert _arena_tickets_zero()
            ran += 1
        else:
            assert entry.plan is None and entry.det_plan is not None, (key, T)
            assert torch.equal(out, want), (key, T)
    assert ran


def test_gpu_k1_152_sm_one_split_ranges_run_at_their_edges(no_k1_dsm):
    """Unforced (no split count), on a part of exactly the SM count of a table of
    :data:`K1_ONE_SPLIT_RANGES_BY_SM` (152-SM ``sm_100a`` / ``sm_103a`` parts): at the first and the last token count
    of every range of a row and at 1025 tokens (an odd number of 64-token K blocks) the prepared launch runs the range's
    one-split program (one chain; by kernel name) and its output meets the deterministic package's budget against FP64
    in each of ``RERUNS`` launches into a NaN-poisoned ``out``; the convenience path binds the same program and its
    output meets that budget too.  One token below and one above a range the row takes the route the planner gives it
    without the range (another program or the deterministic program), within the same budget.  The cluster ranges of
    :data:`K1_DSM_RANGES_BY_SM`, which come before the one-split ranges and hold some of these token counts on
    ``sm_100a``, are taken out (``test_gpu_k1_dsm_ranges_run_at_their_edges`` holds them).  A part of any other
    SM count skips."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rows = cake_backend.K1_ONE_SPLIT_RANGES_BY_SM.get((arch, sm_count), {})
    if not rows:
        pytest.skip(f"no K1 one-split range of {arch} on {sm_count} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 97)
    ran = 0
    for key, ranges in sorted(rows.items()):
        _, L, M, N, dt = key
        f32 = dt == "f32"
        out_dtype = torch.float32 if f32 else torch.bfloat16
        for lo, hi, block_n in ranges:
            for T in sorted({lo - 1, lo, 1025, hi, hi + 1}):
                what = f"{M} {N} {dt} {T}"
                G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
                ref = G.double().t() @ X.double()
                out = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
                plan = cake_backend.k1_plan(
                    arch=arch, L=L, M=M, N=N, K=T, out_f32=f32, sm_count=sm_count
                )
                prepared = (
                    cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                        G, X, out
                    )
                )
                if not lo <= T <= hi:
                    assert plan != (block_n, 1), what
                    atomic = isinstance(
                        prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
                    )
                    assert atomic == (plan is not None), what
                    if atomic:
                        assert (prepared.plan.block_n, prepared.plan.splits) == plan
                    prepared.launch()
                    torch.cuda.synchronize()
                    _k1_check(out, ref, what)
                    continue
                assert plan == (block_n, 1), what
                assert isinstance(
                    prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
                ), what
                assert (prepared.plan.block_n, prepared.plan.splits) == plan, what
                assert prepared.plan.chains == 1, what
                assert prepared.template == cake_backend.atomic_k1_template(
                    block_n, f32
                ), what
                _assert_ran(prepared.launch, prepared.module_name)
                torch.cuda.synchronize()
                _k1_check(out, ref, what)
                for i in range(RERUNS):
                    out.fill_(float("nan"))
                    prepared.launch()
                    torch.cuda.synchronize()
                    _k1_check(out, ref, f"{what} run {i}")
                cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                conv = torch.full_like(out, float("nan"))
                assert (
                    cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(
                        G, X, conv
                    )
                    is conv
                )
                torch.cuda.synchronize()
                (entry,) = list(cake_backend._K1_ATOMIC_CACHE.values())
                assert entry.k1 is not None and entry.plan is not None, what
                assert (entry.plan.block_n, entry.plan.splits) == plan, what
                _k1_check(conv, ref, f"{what} convenience")
                ran += 1
    print(
        f"K1_ONE_SPLIT_RANGES {arch} sm_count={sm_count}: {ran} token counts ran the one-split program"
    )
    assert ran


def test_gpu_k1_dsm_ranges_run_at_their_edges():
    """Unforced (no split count), on a part of exactly the SM count of a table of
    :data:`K1_DSM_RANGES_BY_SM` (148-SM and 152-SM ``sm_100a``, 152-SM ``sm_103a`` parts): at the first and the last
    token count of every range of a row the prepared launch runs the range's cluster program (by kernel name) and its
    output meets the deterministic package's FP32 budget against FP64 (:func:`_k1_check`: relative Frobenius error at
    most 5e-5, max error at most 1e-3 times the largest reference element) with the same bits in each of ``RERUNS``
    launches into a NaN-poisoned ``out``; the convenience path binds the same program, with no arena, and its output
    has those bits too.  One token below a
    row's first range, one above its last and next to a gap between two of its ranges the row takes the route the
    planner gives it without the table (another program or the deterministic program), within the same budget; an
    edge with an odd number of 64-token K blocks inside a range of :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` runs that
    table's program within the same budget: a cluster program of the table's tile and split count (by kernel name),
    or no cluster program.  A part of any other SM count or architecture skips."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rows = cake_backend.K1_DSM_RANGES_BY_SM.get((arch, sm_count), {})
    if not rows:
        pytest.skip(f"no K1 cluster range of {arch} on {sm_count} SMs")
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    g = torch.Generator(device=device).manual_seed(SEED + 98)
    ran = 0
    for key, ranges in sorted(rows.items()):
        _, L, M, N, dt = key
        assert dt == "f32", key
        edges = {t for lo, hi, _ in ranges for t in (lo, hi)}
        outside = {ranges[0][0] - 1, ranges[-1][1] + 1}
        for (_, hi, _), (lo, _, _) in zip(ranges[:-1], ranges[1:], strict=True):
            if lo > hi + 1:
                outside |= {hi + 1, lo - 1}
        for T in sorted(edges | outside):
            what = f"{M} {N} {dt} {T}"
            held = [v for lo, hi, v in ranges if lo <= T <= hi]
            G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
            ref = G.double().t() @ X.double()
            out = torch.full((N, M), float("nan"), device=device, dtype=torch.float32)
            dsm = cake_backend.k1_dsm_plan(
                arch=arch, L=L, M=M, N=N, K=T, out_f32=True, sm_count=sm_count
            )
            prepared = (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                    G, X, out
                )
            )
            atomic = isinstance(
                prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch
            )
            odd = cake_backend.k1_dsm_odd_block_plan(
                arch=arch, L=L, M=M, N=N, K=T, out_f32=True, sm_count=sm_count
            )
            if not held or odd is not None:
                # outside the ranges: no cluster program; an odd number of K blocks inside a range of
                # K1_DSM_ODD_BLOCK_RANGES_BY_SM: that table's program (a cluster program of its tile and split count,
                # or no cluster program)
                assert (dsm is None) == (not held), what
                odd_dsm = bool(held) and odd is not None and len(odd) == 2
                assert bool(atomic and prepared.plan.dsm) == odd_dsm, what
                if odd_dsm:
                    assert (prepared.plan.block_n, prepared.plan.splits) == odd, what
                    assert prepared.template == cake_backend.dsm_k1_template(*odd), what
                    _assert_ran(prepared.launch, prepared.module_name)
                else:
                    prepared.launch()
                torch.cuda.synchronize()
                _k1_check(out, ref, what)
                continue
            assert dsm == held[0], what
            assert atomic and prepared.plan.dsm, what
            assert (prepared.plan.block_n, prepared.plan.splits) == dsm, what
            assert prepared.template == cake_backend.dsm_k1_template(*dsm), what
            _assert_ran(prepared.launch, prepared.module_name)
            torch.cuda.synchronize()
            _k1_check(out, ref, what)
            first = out.clone()
            for i in range(RERUNS):
                out.fill_(float("nan"))
                prepared.launch()
                torch.cuda.synchronize()
                _k1_check(out, ref, f"{what} run {i}")
                assert torch.equal(out, first), f"{what} run {i}: other bits"
            cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
            conv = torch.full_like(out, float("nan"))
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, conv)
                is conv
            )
            torch.cuda.synchronize()
            (entry,) = list(cake_backend._K1_ATOMIC_CACHE.values())
            assert entry.k1 is not None and entry.plan is not None, what
            assert entry.plan.dsm and entry.arena is None, what
            assert (entry.plan.block_n, entry.plan.splits) == dsm, what
            _k1_check(conv, ref, f"{what} convenience")
            assert torch.equal(conv, first), f"{what} convenience: other bits"
            ran += 1
    print(
        f"K1_DSM_RANGES {arch} sm_count={sm_count}: {ran} token counts ran the cluster program"
    )
    assert ran


def test_gpu_router_token_windows_run_at_their_edges():
    """Unforced (no split count): at the first and the last token count of every token window
    of a router weight-gradient row of this architecture (the smallest measured count, 64, for a window other than
    SIMT that starts at one token; a SIMT window runs from its first token count) the prepared launch runs the
    program the planner gives -- the SIMT program of a SIMT window and
    the partition of a head/tail window on the measured SM count, a "waves" window's partition where
    :func:`cake_backend.router_waves` gives one and no SIMT window holds the token count -- and meets the gate; one
    token outside every window, or on another SM count, it runs the deterministic launch."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rules, measured = cake_backend.router_rules(arch, sm_count)
    atomic = cake_backend.CakeDenseGemmNondetRouterLaunch
    ran = 0
    for (rule_arch, layout, hidden, _experts), rule in sorted(
        rules.items(), key=lambda kv: kv[0]
    ):
        if rule_arch != arch or layout != "nn_t" or not isinstance(rule, tuple):
            continue
        for first, last, spec in rule:
            if cake_backend.router_det_spec_splits(spec) is not None:
                continue  # test_gpu_router_det_split_rows_run_the_det_program
            if cake_backend.router_simt_tile(spec) is None:
                first_run = max(first, 64)
            else:
                first_run = first
            for T, inside in (
                (first - 1, False),
                (first_run, True),
                (last, True),
                (last + 1, False),
            ):
                if T < 1:
                    continue
                X, W, G = _router_inputs(T, device, SEED + 92, K=hidden)
                A, B, out = _router_case("wgrad", X, W, G)
                part = cake_backend.router_partition(
                    "nn_t", arch=arch, sm_count=sm_count, M=hidden, N=256, K=T
                )
                simt = cake_backend.router_simt(
                    "nn_t", arch=arch, sm_count=sm_count, M=hidden, N=256, K=T
                )
                prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
                    A, B, out
                )
                tile = cake_backend.router_simt_tile(spec)
                if not inside:
                    assert part is None, (hidden, T)
                elif tile is not None:
                    assert simt == (tile if measured else None), (hidden, T)
                elif spec != "waves" and measured:
                    assert part is not None, (hidden, T)
                if simt is not None:
                    assert part is None, (hidden, T)
                    assert isinstance(prepared, atomic), (hidden, T)
                    assert tuple(prepared.plan.simt) == simt, (hidden, T)
                elif part is None:
                    assert not isinstance(prepared, atomic), (hidden, T)
                    continue
                else:
                    assert isinstance(prepared, atomic), (hidden, T)
                    plan = prepared.plan
                    got = (plan.units, plan.head_spans, plan.head_steps)
                    assert got == part, (hidden, T)
                _assert_ran(prepared.launch, prepared.module_name)
                torch.cuda.synchronize()
                assert _rel_fro(out, A.double() @ B.double()) <= ROUTER_REL_FRO_MAX
                ran += 1
    print(f"ROUTER_WINDOW_EDGES {arch} sm_count={sm_count}: {ran} launches ran")
    if not ran:
        pytest.skip(f"no router token window of {arch} applies on {sm_count} SMs")


def test_gpu_router_rows_without_a_verdict_run_the_deterministic_program_unplanned():
    """Without a split count on a router row without a verdict (the forward; the weight
    gradient at a token count outside every window): the deterministic program's bits (its default split counts),
    through this package's binding of it (``_K2_SPLITK_CACHE``); this package plans nothing for the call."""
    device, arch = _atomic_device_or_skip()
    for hidden in (4096, 5120, 6144):
        X, W, G = _router_inputs(8000, device, SEED + 23, K=hidden)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        fwd = cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W)
        wgrad = cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X)
        assert not cake_backend._K2_ATOMIC_CACHE
        assert len(cake_backend._K2_SPLITK_CACHE) == 2
        assert torch.equal(
            fwd, det_backend.cake_dense_gemm_det_train_router_forward(X, W)
        )
        assert torch.equal(
            wgrad, det_backend.cake_dense_gemm_det_train_router_wgrad(G, X)
        )
        A, B, out = _router_case("fwd", X, W, G)
        assert cake_backend.cake_dense_gemm_nondet_train_router(A, B, out) is out
        assert torch.equal(out, fwd) and not cake_backend._K2_ATOMIC_CACHE


def test_gpu_router_split_rule_rows_run_the_deterministic_count():
    """A weight-gradient row of the deterministic package's ``ROUTER_SPLIT_RULES`` for this device, at the first and
    the last token count of its window where this package has neither a partition nor a SIMT window: the
    deterministic program at that package's count (its ``router_splits``), or at the count of this package's
    ``"det_s<n>"`` window where one holds (:func:`router_det_splits`), through this package's binding
    (``_K2_SPLITK_CACHE``; no plan in ``_K2_ATOMIC_CACHE``), with the deterministic program's bits at that count on
    repeated calls -- the partials reduced by this package's partial sum where it serves the count, else by
    ``torch.sum``."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    ran = 0
    for (rule_arch, sms, layout, hidden), windows in sorted(
        det_backend.ROUTER_SPLIT_RULES.items()
    ):
        if (rule_arch, sms, layout) != (arch, sm_count, "nn_t"):
            continue
        for lo, hi, splits in windows:
            for T in (lo, hi):
                kw = dict(arch=arch, sm_count=sm_count, M=hidden, N=256, K=T)
                if (
                    cake_backend.router_partition("nn_t", **kw) is not None
                    or cake_backend.router_simt("nn_t", **kw) is not None
                ):
                    continue  # this package's program
                X, _, G = _router_inputs(T, device, SEED + 120, K=hidden)
                cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                own_n = cake_backend.router_det_splits("nn_t", **kw) or splits
                want = det_backend.cake_dense_gemm_det_train_router_wgrad(
                    G, X, splits=own_n
                )
                out = torch.empty(256, hidden, device=device)
                for _ in range(2):
                    out.fill_(float("nan"))
                    assert (
                        cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
                            G, X, out
                        )
                        is out
                    )
                    assert torch.equal(out, want), (hidden, T)
                (entry,) = cake_backend._K2_SPLITK_CACHE.values()
                assert entry.plan.splits == own_n, (hidden, T)
                served = own_n in cake_backend.PARTIALS_SUM_SPLITS
                assert (entry.sum_entry is not None) == served, (hidden, T)
                assert all(
                    e.plan is None for e in cake_backend._K2_ATOMIC_CACHE.values()
                )
                ran += 1
    if not ran:
        pytest.skip(
            f"no ROUTER_SPLIT_RULES weight-gradient row of {arch} runs the deterministic program on {sm_count} SMs"
        )


def test_gpu_router_verdicts_set_at_run_time_are_seen(monkeypatch):
    """The convenience path keeps whether a router row has a verdict (per device and view geometry) and whether any
    router table gives the forward one until a router table changes: on a row without a verdict at the call's token
    count the forward and the weight gradient give the deterministic program's bits (unplanned, also on the second call, from the memo); a
    verdict set at run time in the table of the device's SM count runs this package's program at the next call,
    without clearing a cache, and its removal the deterministic program again."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    table, _ = cake_backend.router_rules(arch, sm_count)
    if table is cake_backend.ROUTER_RULES:
        # a verdict of ROUTER_RULES applies on this device whatever its SM count
        monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
    # no verdict at 1000 tokens on any architecture and SM count (the hidden-6144 head/tail weight-gradient windows
    # start at 16385 tokens on 152-SM sm_103a and at 16172 on 212-SM sm_107a)
    hidden, T = 6144, 1000
    X, W, G = _router_inputs(T, device, SEED + 26, K=hidden)
    keys = {"fwd": (arch, "kk", hidden, 256), "wgrad": (arch, "nn_t", hidden, 256)}
    for layout, M, K in (("kk", T, hidden), ("nn_t", hidden, T)):
        spec = cake_backend._router_rule_spec(
            layout, arch=arch, sm_count=sm_count, M=M, N=256, K=K
        )
        assert spec is None, (layout, spec)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    want_fwd = det_backend.cake_dense_gemm_det_train_router(
        X, W.t(), torch.empty(T, 256, device=device)
    )
    want_wgrad = det_backend.cake_dense_gemm_det_train_router(
        G.t(), X, torch.empty(256, hidden, device=device)
    )

    def unplanned():
        for _ in range(2):
            fwd = cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W)
            wgrad = cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X)
            assert torch.equal(fwd, want_fwd) and torch.equal(wgrad, want_wgrad)
        assert not cake_backend._K2_ATOMIC_CACHE

    unplanned()
    monkeypatch.setitem(table, keys["fwd"], "forward")
    monkeypatch.setitem(table, keys["wgrad"], "pairs")
    ref_fwd, ref_wgrad = X.double() @ W.double().t(), G.double().t() @ X.double()

    def fwd():
        return cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W)

    def wgrad():
        return cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X)

    for op, A, B, ref, call in (
        ("fwd", X, W.t(), ref_fwd, fwd),
        ("wgrad", G.t(), X, ref_wgrad, wgrad),
    ):
        out = torch.empty(A.shape[0], B.shape[1], device=device)
        plan = cake_backend.cake_dense_gemm_nondet_train_router_plan(A, B, out)
        assert isinstance(plan, cake_backend.CakeDenseGemmNondetRouterPlan), op
        _assert_ran(call, cake_jit.select_module(arch, plan.template))
        assert _rel_fro(call(), ref) <= ROUTER_REL_FRO_MAX, op
    first = fwd()
    assert torch.equal(fwd(), first)
    assert len(cake_backend._K2_ATOMIC_CACHE) == 2
    table[keys["fwd"]] = None
    table[keys["wgrad"]] = None
    for _ in range(2):
        assert torch.equal(fwd(), want_fwd) and torch.equal(wgrad(), want_wgrad)


def test_gpu_router_verdict_changed_at_run_time_runs_the_new_program(monkeypatch):
    """A weight-gradient verdict changed from one program to another between calls of the same tensors: the next call
    -- through ``cake_dense_gemm_nondet_train_router_wgrad`` and through ``cake_dense_gemm_nondet_train_router`` -- runs the program the planner gives under the new
    verdict (the stream-K ``"pairs"`` partition, then a SIMT tile this checkout registers for the architecture, else
    two units per tile), within the router budget, without clearing a cache."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    table, _ = cake_backend.router_rules(arch, sm_count)
    if table is cake_backend.ROUTER_RULES:
        # a verdict of ROUTER_RULES applies on this device whatever its SM count
        monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
    hidden, T = 6144, 1000
    X, _, G = _router_inputs(T, device, SEED + 28, K=hidden)
    A, B = G.t(), X
    ref = A.double() @ B.double()
    out = torch.empty(256, hidden, device=device)
    simt = next(
        (
            t
            for t in SIMT_TILES
            if cake_backend.router_simt_template(t) in cake_jit.KERNELS[arch]
        ),
        None,
    )
    calls = (
        lambda: cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, out),
        lambda: cake_backend.cake_dense_gemm_nondet_train_router(A, B, out),
    )
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    plans = []
    for spec in ("pairs", "S2" if simt is None else _simt_spec(simt)):
        monkeypatch.setitem(table, (arch, "nn_t", hidden, 256), ((1, T, spec),))
        plan = cake_backend.cake_dense_gemm_nondet_train_router_plan(A, B, out)
        assert isinstance(plan, cake_backend.CakeDenseGemmNondetRouterPlan), spec
        for call in calls:
            _assert_ran(call, cake_jit.select_module(arch, plan.template))
            assert _rel_fro(call(), ref) <= ROUTER_REL_FRO_MAX, spec
        cached = [e.plan for e in cake_backend._K2_ATOMIC_CACHE.values()]
        assert cached and all(p == plan for p in cached), spec
        plans.append(plan)
    assert plans[0] != plans[1]


def test_gpu_router_deterministic_program_through_the_binding():
    """Calls of this package that run the deterministic program -- the forward and the input gradient, the
    weight gradient of a row without a verdict, the generic ``kn`` call -- go through this package's binding
    (``_K2_SPLITK_CACHE``; nothing in ``_K2_ATOMIC_CACHE``) and give the deterministic package's bits: dense and
    padded outputs (one split stores into a padded ``out`` through its flat alias, several splits reduce into it), a
    split source with padded rows (the torch chain), repeated calls; a larger row grows the arena and the smaller
    rows' launches are carved again with the same bits; another stream binds an arena of its own; a binding the
    deterministic package refuses raises its error."""
    device, arch = _atomic_device_or_skip()
    # no router verdict on any architecture from 257 to 16171 tokens (the SIMT windows end at 256 tokens or before)
    hidden = 6144
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()

    def check(T, seed, pad_out=False, pad_g=False):
        X, W, G = _router_inputs(T, device, seed, K=hidden)
        if pad_g:
            G = torch.cat([G, torch.zeros(T, 16, device=device)], 1)[:, :256]
        shapes = {"fwd": (T, 256), "dgrad": (T, hidden), "wgrad": (256, hidden)}
        # op -> (the deterministic package's call, this package's call)
        # fmt: off
        calls = {
            "fwd": (lambda out: det_backend.cake_dense_gemm_det_train_router_forward(X, W, out),
                    lambda out: cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, out)),
            "dgrad": (lambda out: det_backend.cake_dense_gemm_det_train_router_dgrad(G, W, out),
                      lambda out: cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out)),
            "wgrad": (lambda out: det_backend.cake_dense_gemm_det_train_router_wgrad(G, X, out),
                      lambda out: cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, out)),
        }
        # fmt: on
        for op, (det_call, call) in calls.items():
            rows, cols = shapes[op]
            if pad_out:
                storage = torch.full((rows, cols + 64), 7.0, device=device)
                out = storage[:, :cols]
            else:
                out = torch.empty(rows, cols, device=device)
            want = det_call(out).clone()
            for _ in range(2):
                out.fill_(float("nan"))
                assert call(out) is out
                assert torch.equal(out, want), (op, T, pad_out, pad_g)
            if pad_out:
                assert bool((storage[:, cols:] == 7.0).all()), op
        out = torch.empty(T, hidden, device=device)
        want = det_backend.cake_dense_gemm_det_train_router(G, W, out.clone())
        assert cake_backend.cake_dense_gemm_nondet_train_router(G, W, out) is out
        assert torch.equal(out, want)

    variants = ((320, False, False), (1000, True, False), (1000, False, True))
    for T, pad_out, pad_g in variants:
        check(T, SEED + 70 + T, pad_out, pad_g)
    assert cake_backend._K2_SPLITK_CACHE and not cake_backend._K2_ATOMIC_CACHE
    entries = list(cake_backend._K2_SPLITK_CACHE.values())
    assert all(e.k2 is not None for e in entries)
    assert any(e.k2.partials_shape is not None for e in entries)
    assert any(e.k2.partials_shape is None for e in entries)
    # several splits: this package's partial sum reduces the partials (the forward's four, the weight gradient's six)
    assert all(
        (e.sum_entry is not None) == (e.k2.partials_shape is not None) for e in entries
    )
    assert any(e.split_entry is None for e in entries)  # the padded split source
    versions = {id(e): e.k2.version for e in entries}
    check(8000, SEED + 71)  # a larger row: the arena grows
    check(320, SEED + 70 + 320)
    assert any(e.k2.version != versions[id(e)] for e in entries)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        check(1000, SEED + 72)
    torch.cuda.synchronize()
    assert stream.cuda_stream in {key[1] for key in cake_backend._ATOMIC_ARENAS}
    # a binding the deterministic package refuses: its error
    X, W, G = _router_inputs(1001, device, SEED + 73, K=hidden)
    storage = torch.empty(1001 * hidden + 4, device=device)
    out = storage[1 : 1 + 1001 * hidden].view(1001, hidden)
    with pytest.raises(ValueError, match="16-byte aligned") as refused:
        det_backend.cake_dense_gemm_det_train_router_dgrad(G, W, out)
    with pytest.raises(ValueError, match=re.escape(str(refused.value))):
        cake_backend.cake_dense_gemm_nondet_train_router_dgrad(G, W, out)


def _special_partials(splits, rows, cols, seed):
    """FP32 partials ``[splits, rows, cols]`` (on the CPU) with signed zeros, NaN, infinities, subnormals, values whose
    sums overflow, whole columns of signed zeros and exact cancellations next to scaled normal values."""
    g = torch.Generator().manual_seed(seed)
    n = splits * rows * cols
    base = torch.randn(n, generator=g) * torch.pow(
        10.0, torch.randint(-3, 4, (n,), generator=g).float()
    )
    kind = torch.randint(0, 20, (n,), generator=g)
    v = base.clone()
    v[kind == 0] = 0.0
    v[kind == 1] = -0.0
    v[kind == 2] = float("nan")
    v[kind == 3] = float("inf")
    v[kind == 4] = float("-inf")
    v[kind == 5] = 1e-40 * torch.sign(base[kind == 5])
    v[kind == 6] = 3.0e38 * torch.sign(base[kind == 6])
    v = v.view(splits, rows * cols)
    picked = torch.randint(0, rows * cols, (max(3, rows * cols // 16),), generator=g)
    third = len(picked) // 3
    v[:, picked[:third]] = -0.0
    v[:, picked[third : 2 * third]] = torch.where(
        torch.rand(splits, 2 * third - third, generator=g) < 0.5, 0.0, -0.0
    )
    cancel = picked[2 * third :]
    v[1, cancel] = -v[0, cancel]
    v[2:, cancel] = -0.0
    return v.view(splits, rows, cols)


def _partials_sum_in_written_order(partials):
    """The partial sum's written order: four accumulators from ``+0``, partial ``s`` into accumulator ``s % 4`` in
    ascending ``s``, then ``((a0 + a1) + a2) + a3`` (one FP32 add per step)."""
    acc = [torch.zeros_like(partials[0]) for _ in range(4)]
    for s in range(partials.shape[0]):
        acc[s % 4] = acc[s % 4] + partials[s]
    return ((acc[0] + acc[1]) + acc[2]) + acc[3]


def _launch_partials_sum(partials, out, pdl):
    """Launch this package's partial sum of ``partials`` into ``out`` on the current stream, bound as the binding of
    the deterministic K2 program binds it (``out`` through its flat alias when padded)."""
    index = out.get_device()
    arch = det_backend.arch_for(out.device)
    entry, template = cake_backend._partials_sum_binding(
        arch, tuple(partials.shape), out, pdl
    )
    out_arg = out if out.is_contiguous() else det_backend.flat_alias(out)
    args = [
        partials
        if a is cake_backend._K2_PARTIALS_ARG
        else out_arg
        if a is cake_backend._K2_CALL_ARGS[1]
        else a
        for a in template
    ]
    with det_backend._ffi_stream_context(index):
        entry(*args)


def test_gpu_partials_sum_is_torch_sum_and_the_written_order():
    """This package's partial sum gives the bits of ``torch.sum(partials, 0, out=out)`` -- the deterministic
    package's reduction -- and of its written order, on every split count it serves (two to eight), with signed
    zeros, NaN, infinities, subnormals, overflowing sums and exact cancellations: every 32-bit word of ``out``
    compared.  Rows of a column count that is no multiple of four, partials and ``out`` at an address that is no
    multiple of 16 bytes (the scalar loads and stores) and a padded ``out`` (row stride above the columns, the padding
    untouched) included, with and without PDL."""
    device, arch = _atomic_device_or_skip()
    _require_template(arch, cake_backend.PARTIALS_SUM_TEMPLATE)
    shapes = ((1, 8), (3, 37), (64, 256), (1000, 32), (129, 1024))
    checked = 0
    for splits in cake_backend.PARTIALS_SUM_SPLITS:
        for rows, cols in shapes:
            host = _special_partials(splits, rows, cols, SEED + 100 * splits + cols)
            for ws_offset, out_kind, pdl in (
                (0, "dense", 0),
                (1, "dense", 1),
                (0, "padded", 1),
                (1, "offset", 0),
            ):
                n = splits * rows * cols
                ws_buf = torch.empty(n + 8, device=device)
                partials = ws_buf[ws_offset : ws_offset + n].view(splits, rows, cols)
                partials.copy_(host.to(device))
                if out_kind == "padded":
                    storage = torch.full((rows, cols + 4), 7.0, device=device)
                    out = storage[:, :cols]
                elif out_kind == "offset":
                    storage = torch.full((rows * cols + 4,), 7.0, device=device)
                    out = storage[1 : 1 + rows * cols].view(rows, cols)
                else:
                    storage = None
                    out = torch.empty(rows, cols, device=device)
                assert cake_backend._partials_sum_serves(partials.shape, out)
                want = torch.full((rows, cols), float("nan"), device=device)
                torch.sum(partials, 0, out=want)
                written = _partials_sum_in_written_order(partials)
                out.fill_(float("nan"))
                _launch_partials_sum(partials, out, pdl)
                torch.cuda.synchronize()
                case = (splits, rows, cols, ws_offset, out_kind, pdl)
                bits = out.view(torch.int32)
                assert torch.equal(bits, want.view(torch.int32)), case
                assert torch.equal(bits, written.view(torch.int32)), case
                if out_kind == "padded":
                    assert bool((storage[:, cols:] == 7.0).all()), case
                elif out_kind == "offset":
                    assert float(storage[0]) == 7.0 and float(storage[-1]) == 7.0, case
                checked += 1
    print(
        f"PARTIALS_SUM_BITS {arch}: {checked} launches equal torch.sum and the written order"
    )


def test_gpu_router_partials_sum_eager_graph_and_recompute():
    """The deterministic router program through this package's binding with its partial sum (a row without a router
    verdict: the forward's four splits, the weight gradient's six) gives the deterministic package's bits -- its
    ``torch.sum`` -- eagerly, again when recomputed (a checkpointed forward runs it twice) and from a CUDA graph that
    captured the calls, replayed several times."""
    device, arch = _atomic_device_or_skip()
    _require_template(arch, cake_backend.PARTIALS_SUM_TEMPLATE)
    hidden, T = 6144, 4096  # no router verdict at 4096 tokens on any architecture
    X, W, G = _router_inputs(T, device, SEED + 91, K=hidden)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    want_fwd = det_backend.cake_dense_gemm_det_train_router_forward(
        X, W, enable_pdl=False
    )
    want_wgrad = det_backend.cake_dense_gemm_det_train_router_wgrad(
        G, X, enable_pdl=False
    )
    Y = torch.empty(T, 256, device=device)
    dW = torch.empty(256, hidden, device=device)
    stream = torch.cuda.Stream(device)

    def step():
        cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, Y)
        cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, dW)

    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        step()
    stream.synchronize()
    entries = [
        e
        for key, e in cake_backend._K2_SPLITK_CACHE.items()
        if key[1] == stream.cuda_stream
    ]
    assert sorted(e.plan.splits for e in entries) == [4, 6]
    assert all(e.sum_entry is not None for e in entries)
    assert not cake_backend._K2_ATOMIC_CACHE
    assert torch.equal(Y, want_fwd) and torch.equal(dW, want_wgrad)
    eager = (Y.clone(), dW.clone())
    for t in (Y, dW):
        t.fill_(float("nan"))
    # the fill runs on the current stream: the recompute's stores must come after it
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        step()  # the recompute
    stream.synchronize()
    assert torch.equal(Y, eager[0]) and torch.equal(dW, eager[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    for _ in range(3):
        for t in (Y, dW):
            t.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(Y, eager[0]) and torch.equal(dW, eager[1])


def test_gpu_router_binding_pdl_waits_for_an_early_producer():
    """The deterministic router program through this package's binding as programmatic dependent launches: the
    forward writes ``Y``, the input gradient reads it and writes ``dX``, the weight gradient (a row without a verdict)
    reads both; each starts while its predecessor runs and must still read every value it wrote.  The flag is bound
    into the cached argument lists (a key per flag), and every launch gives the deterministic package's bits."""
    device, arch = _atomic_device_or_skip()
    hidden, T = 6144, 4096  # no router verdict at 4096 tokens on any architecture
    X, W, _ = _router_inputs(T, device, SEED + 80, K=hidden)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    want_y = det_backend.cake_dense_gemm_det_train_router_forward(
        X, W, enable_pdl=False
    )
    want_dx = det_backend.cake_dense_gemm_det_train_router_dgrad(
        want_y, W, enable_pdl=False
    )
    want_dw = det_backend.cake_dense_gemm_det_train_router_wgrad(
        want_y, want_dx, enable_pdl=False
    )
    Y = torch.empty(T, 256, device=device)
    dX = torch.empty(T, hidden, device=device)
    dW = torch.empty(256, hidden, device=device)
    for pdl in (True, False, None):
        for _ in range(3):
            for t in (Y, dX, dW):
                t.fill_(float("nan"))
            kw = dict(enable_pdl=pdl)
            cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, Y, **kw)
            cake_backend.cake_dense_gemm_nondet_train_router_dgrad(Y, W, dX, **kw)
            cake_backend.cake_dense_gemm_nondet_train_router_wgrad(Y, dX, dW, **kw)
            torch.cuda.synchronize()
            assert torch.equal(Y, want_y) and torch.equal(dX, want_dx), pdl
            assert torch.equal(dW, want_dw), pdl
    assert not cake_backend._K2_ATOMIC_CACHE
    flags = {key[2][1][1] for key in cake_backend._K2_SPLITK_CACHE}
    assert flags == {0, 1}
    assert len(cake_backend._K2_SPLITK_CACHE) == 6


def test_gpu_cached_router_launch_is_the_prepared_launch():
    """The convenience path's bound launch (arguments ordered once, the stack converted once per arena growth) runs
    the prepared launch's program on the same arguments: a reproducible plan gives its bits, every plan meets the
    gate; after the arena of the stream grows (a larger row) the launch is bound again and still gives them."""
    device, arch = _atomic_device_or_skip()
    cases = []
    for T, splits in ((64, None), (128, None), (192, None), (1024, 3), (300, 2)):
        X, W, G = _router_inputs(T, device, SEED + 24, K=5120)
        cases.append((_router_case("fwd" if splits == 2 else "wgrad", X, W, G), splits))
    ran = 0
    for grown in (False, True):
        if grown:
            X, W, G = _router_inputs(4096, device, SEED + 25, K=5120)
            A, B, out = _router_case("wgrad", X, W, G)
            cake_backend.cake_dense_gemm_nondet_train_router(A, B, out, splits=4)
        for (A, B, out), splits in cases:
            plan = cake_backend.cake_dense_gemm_nondet_train_router_plan(
                A, B, out, splits=splits
            )
            if not isinstance(plan, cake_backend.CakeDenseGemmNondetRouterPlan):
                continue
            prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
                A, B, out.clone(), splits=splits
            )
            want = prepared.launch().clone()
            got = cake_backend.cake_dense_gemm_nondet_train_router(
                A, B, out, splits=splits
            )
            torch.cuda.synchronize()
            ref = A.double() @ B.double()
            assert _rel_fro(got, ref) <= ROUTER_REL_FRO_MAX
            if plan.reproducible:
                assert torch.equal(got, want), (tuple(A.shape), splits)
            ran += 1
    print(f"ROUTER_BOUND_LAUNCH {arch}: {ran} launches")
    assert ran


def test_gpu_cached_router_launch_runs_the_head_tail_windows_at_their_edges(
    monkeypatch,
):
    """Unforced (no split count): at the first and the last token count of every head/tail
    window of this device's router table (``router_rules`` of its architecture and SM count), the convenience path's
    cached launch (its bound arguments, ``_K2Binding``) runs the prepared launch's plan and program, meets the gate
    and, for a reproducible plan, gives the prepared launch's bits; the second call of the key reuses the binding.
    The shipped tables hold no head/tail window (those windows varied from run to run and missed the accuracy check);
    the former hidden-5120 window at 16172 to 16384 tokens (``"H0.5x4"`` on ``sm_107a``, ``"H0.5x3"`` elsewhere) is
    set here in the device's table in place of the device's own windows over those token counts (on 212-SM
    ``sm_107a`` parts the ``"det_s12"`` window, which comes first), so the registered head/tail program keeps its
    coverage."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rules, holds = cake_backend.router_rules(arch, sm_count)
    key = (arch, "nn_t", 5120, 256)
    former = (16172, 16384, "H0.5x4" if arch == "sm_107a" else "H0.5x3")
    others = tuple(
        window
        for window in rules.get(key) or ()
        if window[1] < former[0] or window[0] > former[1]
    )
    monkeypatch.setitem(rules, key, others + (former,))
    edges = []
    for (rule_arch, layout, hidden, _experts), rule in sorted(
        rules.items(), key=lambda kv: kv[0]
    ):
        if rule_arch != arch or layout != "nn_t" or not isinstance(rule, tuple):
            continue
        for first, last, spec in rule:
            if holds and cake_backend.router_head_tail(spec) is not None:
                edges += [(hidden, first), (hidden, last)]
    ran = 0
    for hidden, T in edges:
        X, W, G = _router_inputs(T, device, SEED + 26, K=hidden)
        A, B, out = _router_case("wgrad", X, W, G)
        prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
            A, B, out.clone()
        )
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch), (
            hidden,
            T,
        )
        assert prepared.plan.head_spans > 0, (hidden, T)
        want = prepared.launch().clone()
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        _assert_ran(
            lambda A=A, B=B, out=out: cake_backend.cake_dense_gemm_nondet_train_router(
                A, B, out
            ),
            prepared.module_name,
        )
        torch.cuda.synchronize()
        (entry,) = cake_backend._K2_ATOMIC_CACHE.values()
        assert entry.plan == prepared.plan, (hidden, T)
        assert entry.module_name == prepared.module_name, (hidden, T)
        bound = entry.k2
        assert bound is not None and bound.version == entry.arena.version
        ref = A.double() @ B.double()
        assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX, (hidden, T)
        assert _rel_fro(want, ref) <= ROUTER_REL_FRO_MAX, (hidden, T)
        if prepared.plan.reproducible:
            assert torch.equal(out, want), (hidden, T)
        out.fill_(float("nan"))
        assert cake_backend.cake_dense_gemm_nondet_train_router(A, B, out) is out
        torch.cuda.synchronize()
        (again,) = cake_backend._K2_ATOMIC_CACHE.values()
        assert again is entry and again.k2 is bound, (hidden, T)
        assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX, (hidden, T)
        if prepared.plan.reproducible:
            assert torch.equal(out, want), (hidden, T)
        ran += 1
    print(f"ROUTER_BOUND_HEAD_TAIL_EDGES {arch} sm_count={sm_count}: {ran} launches")
    if not ran:
        pytest.skip(f"no router head/tail window of {arch} applies on {sm_count} SMs")


@pytest.mark.parametrize("op, splits", [("fwd", 2), ("wgrad", 4)])
def test_gpu_router_padded_out_rows(op, splits):
    """An ``out`` whose rows are padded (row stride above the row length): the prepared launch and the
    convenience path fill and write only ``out``'s elements -- the padding keeps its values -- and meet the gate."""
    device, arch = _atomic_device_or_skip()
    X, W, G = _router_inputs(ROUTER_T, device, SEED + 23)
    A, B, dense = _router_case(op, X, W, G)
    rows, cols = dense.shape
    storage = torch.full((rows, cols + 64), 7.0, device=device)
    out = storage[:, :cols]
    prepared = _router_prepare(A, B, out, splits, arch)
    assert prepared.plan.part_tiles
    ref = A.double() @ B.double()
    for launch in (
        prepared.launch,
        lambda: cake_backend.cake_dense_gemm_nondet_train_router(
            A, B, out, splits=splits
        ),
    ):
        for _ in range(3):
            out.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
            assert bool(torch.isfinite(out).all())
            assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX
            assert bool((storage[:, cols:] == 7.0).all())


def test_gpu_router_misaligned_out_takes_the_deterministic_path():
    """An ``out`` 4 bytes off a 16-byte boundary: the call of this package runs the deterministic path, so it
    refuses the single-split row-major forward exactly as the deterministic package does."""
    device, arch = _atomic_device_or_skip()
    X, W, _ = _router_inputs(1001, device, SEED + 22)
    storage = torch.empty(1001 * 256 + 4, device=device)
    out = storage[1 : 1 + 1001 * 256].view(1001, 256)
    with pytest.raises(ValueError, match="16-byte aligned") as refused:
        det_backend.cake_dense_gemm_det_train_router_plan(X, W.t(), out, splits=1)
    message = re.escape(str(refused.value))
    with pytest.raises(ValueError, match=message):
        cake_backend.cake_dense_gemm_nondet_train_router_plan(X, W.t(), out, splits=1)
    with pytest.raises(ValueError, match=message):
        cake_backend.cake_dense_gemm_nondet_train_router_prepare(
            X, W.t(), out, splits=1
        )
    with pytest.raises(ValueError, match=message):
        cake_backend.cake_dense_gemm_nondet_train_router(X, W.t(), out, splits=1)


def _router_entry_of(op, P, Q, out, splits, stream):
    """The ``_K2_ENTRY_CACHE`` entry of a router entry point's call (``op`` and its tensors ``P``, ``Q``, ``out``)
    on ``stream`` with the device's default PDL flag, or ``None``."""
    index = out.get_device()
    pdl = int(det_backend.pdl_enabled(None, index))
    key = cake_backend._k1_key(index, stream.cuda_stream, (op, splits, pdl), P, Q, out)
    return cake_backend._K2_ENTRY_CACHE.get(key)


def _router_entry_bits(entry) -> bool:
    """Whether a cached router entry's launch repeats bit for bit: the deterministic program, or a reproducible plan
    of this package's."""
    return not entry.atomic or entry.launch.plan.reproducible


@pytest.mark.parametrize("hidden, T", [(5120, 64), (4096, 256), (6144, 1000)])
def test_gpu_router_entry_points_give_their_paths_results(monkeypatch, hidden, T):
    """Through ``_K2_ENTRY_CACHE``: every router entry point's call -- the forward with and
    without a split count, the input gradient, the weight gradient with and without one; dense and padded outputs;
    another stream -- binds the launch its own path takes and gives that path's results on repeated calls: the
    deterministic program and a reproducible plan bit for bit (the deterministic program the deterministic package's
    bits), every launch within the gate; the padding keeps its values."""
    device, arch = _atomic_device_or_skip()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    X, W, G = _router_inputs(T, device, SEED + 100 + T, K=hidden)
    tensors = {"fwd": (X, W), "dgrad": (G, W), "wgrad": (G, X)}
    calls = {
        "fwd": cake_backend.cake_dense_gemm_nondet_train_router_forward,
        "dgrad": cake_backend.cake_dense_gemm_nondet_train_router_dgrad,
        "wgrad": cake_backend.cake_dense_gemm_nondet_train_router_wgrad,
    }
    det_calls = {
        "fwd": det_backend.cake_dense_gemm_det_train_router_forward,
        "dgrad": det_backend.cake_dense_gemm_det_train_router_dgrad,
        "wgrad": det_backend.cake_dense_gemm_det_train_router_wgrad,
    }
    shapes = {"fwd": (T, 256), "dgrad": (T, hidden), "wgrad": (256, hidden)}
    side = torch.cuda.Stream(device)
    ran = {"bits": 0, "gate": 0}
    # fmt: off
    cases = (
        ("fwd", None, False, False), ("fwd", 2, False, False), ("fwd", None, True, True),
        ("dgrad", None, False, False), ("dgrad", None, True, True),
        ("wgrad", None, False, False), ("wgrad", 2, True, False), ("wgrad", None, False, True),
    )
    # fmt: on
    for op, splits, pad, on_side in cases:
        P, Q = tensors[op]
        kw = dict()
        if op != "dgrad":
            kw["splits"] = splits
        rows, cols = shapes[op]
        storage = torch.full((rows, cols + (64 if pad else 0)), 7.0, device=device)
        out = storage[:, :cols]
        ref = (P.double().t() if op == "wgrad" else P.double()) @ (
            Q.double().t() if op == "fwd" else Q.double()
        )
        stream = side if on_side else torch.cuda.current_stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            with monkeypatch.context() as m:
                m.setattr(cake_backend, "_k2_entry", lambda *args: None)
                want = calls[op](P, Q, out, **kw).clone()
            for _ in range(3):
                out.fill_(float("nan"))
                assert calls[op](P, Q, out, **kw) is out
                stream.synchronize()
                entry = _router_entry_of(op, P, Q, out, splits, stream)
                assert entry is not None and entry.launch is not None, (op, splits, pad)
                assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX, (op, splits, pad)
                assert bool((storage[:, cols:] == 7.0).all()), (op, splits, pad)
                if _router_entry_bits(entry):
                    assert torch.equal(out, want), (op, splits, pad)
            if not entry.atomic:
                det_kw = {} if op == "dgrad" or splits is None else {"splits": splits}
                det_out = det_calls[op](P, Q, torch.empty_like(out), **det_kw)
                stream.synchronize()
                assert torch.equal(out, det_out), (op, splits, pad)
        ran["bits" if _router_entry_bits(entry) else "gate"] += 1
    print(f"ROUTER_ENTRY {arch} H{hidden} T{T}: {ran}")


def test_gpu_router_entry_points_capture_in_a_cuda_graph(monkeypatch):
    """The router entry points' calls through ``_K2_ENTRY_CACHE`` (the forward with and
    without a split count, the input gradient, the weight gradient of a row with a verdict and of one without), their
    keys cached on the capturing stream, capture in a CUDA graph without planning: every replay gives the eager calls'
    results -- bit for bit for the deterministic program and a reproducible plan -- within the gate."""
    device, arch = _atomic_device_or_skip()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    stream = torch.cuda.Stream(device)
    cases = []
    for hidden, T in ((5120, 64), (6144, 1000)):
        X, W, G = _router_inputs(T, device, SEED + 110 + T, K=hidden)
        # name -> (op, P, Q, splits, out)
        # fmt: off
        cases += [
            ("fwd", X, W, None, torch.empty(T, 256, device=device)),
            ("fwd", X, W, 2, torch.empty(T, 256, device=device)),
            ("dgrad", G, W, None, torch.empty(T, hidden, device=device)),
            ("wgrad", G, X, None, torch.empty(256, hidden, device=device)),
        ]
        # fmt: on

    def run_all():
        for op, P, Q, splits, out in cases:
            if op == "fwd":
                cake_backend.cake_dense_gemm_nondet_train_router_forward(
                    P, Q, out, splits=splits
                )
            elif op == "dgrad":
                cake_backend.cake_dense_gemm_nondet_train_router_dgrad(P, Q, out)
            else:
                cake_backend.cake_dense_gemm_nondet_train_router_wgrad(
                    P, Q, out, splits=splits
                )

    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(2):
            run_all()
    stream.synchronize()
    eager = [out.clone() for *_, out in cases]
    entries = [_router_entry_of(op, P, Q, out, s, stream) for op, P, Q, s, out in cases]
    assert all(e is not None and e.launch is not None for e in entries)
    keys = set(cake_backend._K2_ENTRY_CACHE)
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as m:
        m.setattr(
            cake_backend,
            "_plan_k2_entry",
            lambda *args: pytest.fail("planned in capture"),
        )
        with torch.cuda.graph(graph, stream=stream):
            run_all()
    assert set(cake_backend._K2_ENTRY_CACHE) == keys
    for _ in range(3):
        for *_, out in cases:
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for (op, P, Q, splits, out), want, entry in zip(
            cases, eager, entries, strict=True
        ):
            ref = (P.double().t() if op == "wgrad" else P.double()) @ (
                Q.double().t() if op == "fwd" else Q.double()
            )
            assert _rel_fro(out, ref) <= ROUTER_REL_FRO_MAX, (op, splits)
            if _router_entry_bits(entry):
                assert torch.equal(out, want), (op, splits)
    print(
        f"ROUTER_ENTRY_GRAPH {arch}: {len(cases)} launches, "
        f"{sum(map(_router_entry_bits, entries))} bit for bit"
    )


# ---------------------------------------------------------------------------
# GPU: streams, graphs, allocation-free prepared launches
# ---------------------------------------------------------------------------


def test_gpu_concurrent_streams_keep_their_own_workspaces(k1_rule, router_rule):
    """Two streams issuing K1 and K2 calls of this package of different geometries concurrently: one arena per
    (device, stream), every result within budget, the tickets zero."""
    device, arch = _atomic_device_or_skip()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    streams = [torch.cuda.Stream(device) for _ in range(2)]
    cases = []
    for i, (row, T) in enumerate((("indexer_k", 16231), ("thin_48", 4097))):
        G, X, out = _k1_wgrad_inputs(row, T, torch.bfloat16, device, SEED + 60 + i)
        cases.append((G, X, out))
    k1_rule(*cases[0], 2)
    k1_rule(*cases[1], 3)
    Xf, Wf, Gf = _router_inputs(ROUTER_T, device, SEED + 62)
    router_rule("wgrad", Gf.t(), Xf, "pairs")
    router_outs = [
        torch.empty(Gf.shape[1], Xf.shape[1], device=device) for _ in streams
    ]
    torch.cuda.synchronize()
    for _ in range(5):
        for stream, (G, X, out), rout in zip(streams, cases, router_outs, strict=True):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)
                cake_backend.cake_dense_gemm_nondet_train_router_wgrad(Gf, Xf, rout)
    torch.cuda.synchronize()
    for G, X, out in cases:
        _k1_check(out, G.double().t() @ X.double())
    ref = Gf.double().t() @ Xf.double()
    for rout in router_outs:
        assert _rel_fro(rout, ref) <= ROUTER_REL_FRO_MAX
    keys = {key[1] for key in cake_backend._ATOMIC_ARENAS}
    assert {s.cuda_stream for s in streams} <= keys
    assert _arena_tickets_zero()


def test_gpu_prepared_launches_allocate_nothing_and_capture_into_a_graph():
    """The prepared K1 and K2 launches allocate nothing on the device, capture into one CUDA graph (K2: the split,
    the memset of ``out`` and the kernel) and replay with fresh operand values copied in place: the replay meets the
    budgets, and the K1 tickets are zero after the replays."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 70)
    k1 = _k1_prepare(G, X, out, arch, 3)
    Xf, Wf, Gf = _router_inputs(ROUTER_T, device, SEED + 71)
    A2, B2, out2 = _router_case("wgrad", Xf, Wf, Gf)
    k2 = _router_prepare(A2, B2, out2, 4, arch)
    assert k2.plan.part_tiles
    k1.launch(), k2.launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(5):
        k1.launch(), k2.launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        k1.launch()
        k2.launch()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.Generator(device=device).manual_seed(SEED + 72)
    for _ in range(3):
        G.copy_(_bf16(G.shape, device, g))
        X.copy_(_bf16(X.shape, device, g))
        Xf.copy_(torch.randn(Xf.shape, device=device, generator=g))
        Gf.copy_(torch.randn(Gf.shape, device=device, generator=g) * 1e-3)
        out.fill_(float("nan"))
        out2.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double(), "graph replay")
        assert _rel_fro(out2, A2.double() @ B2.double()) <= ROUTER_REL_FRO_MAX
        assert k1.counters_are_zero()


def test_gpu_prepared_k1_reset_counters():
    """``reset_counters`` zeroes the tickets (the recovery after a launch that did not complete)."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_hw", 4097, torch.bfloat16, device, SEED + 80)
    prepared = _k1_prepare(G, X, out, arch, 2)
    prepared.tensors["counters"].view(torch.int32).fill_(1)
    assert not prepared.counters_are_zero()
    prepared.reset_counters()
    prepared.launch()
    torch.cuda.synchronize()
    _k1_check(out, G.double().t() @ X.double())
    assert prepared.counters_are_zero()


# ---------------------------------------------------------------------------
# CUDA graphs of the convenience path: an arena keeps every buffer a captured launch is bound to
# ---------------------------------------------------------------------------

#: The convenience calls :func:`_graph_safety_scenario` captures, with the arena kinds their launches bind (the scenario
#: asserts them): this package's K1 program (split slots and tickets), the deterministic K1 program through this
#: package's binding (stream-K slabs and slice counters), this package's router program (the BF16x3 stack) and the
#: deterministic router program through this package's binding (the stack and the split-K partials).
GRAPH_SAFETY_KINDS = {
    "k1_wgrad": {"k1_slots", "k1_tickets"},
    "k1_forward": {"k1_det_ws", "k1_det_counters"},
    "k2_wgrad": {"k2_stack"},
    "k2_forward": {"k2_stack", "k2_splitk_partials"},
}
GRAPH_SAFETY_CASES = tuple(GRAPH_SAFETY_KINDS)
#: ``(N, K, (T_small, T_large))`` of the k1_forward case: forwards whose deterministic plan splits K across stream-K
#: units (partial slabs in ``k1_det_ws``) at both token counts on the 148- and 152-SM parts, the large one needing more
#: slabs (the scenario asserts both).
GRAPH_SAFETY_K1_FORWARD = (256, 4096, (256, 1000))
#: Seconds a scenario process may take: a replay on freed workspace can fault, or spin on a counter it reads there.
GRAPH_SAFETY_TIMEOUT = 300
#: The byte the filler tensors of :func:`_graph_safety_scenario` hold.
GRAPH_SAFETY_FILL = 0xA5


def _graph_safety_k1_wgrad(row, T, device, seed, monkeypatch):
    G, X, out = _k1_wgrad_inputs(row, T, torch.bfloat16, device, seed)
    _force_k1_rule(monkeypatch, G, X, out, 2)

    def call(o):
        return cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, o)

    def check(got, want):
        # two splits: one add of two partials, bit for bit the eager result
        assert torch.equal(got, want)
        _k1_check(got, G.double().t() @ X.double(), "graph replay")

    return call, out, check


def _graph_safety_k1_forward(T, N, K, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    X, W = _bf16((T, K), device, g), _bf16((N, K), device, g, K**-0.5)
    out = torch.empty(T, N, device=device, dtype=torch.bfloat16)

    def call(o):
        return cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, o)

    def check(got, want):
        assert torch.equal(got, want)  # the deterministic program's bits

    return call, out, check


def _graph_safety_router(op, T, device, seed, monkeypatch):
    X, W, G = _router_inputs(T, device, seed)
    if op == "wgrad":
        out = torch.empty(W.shape, device=device)
        # a verdict for the row at every token count: this package's stream-K program with its BF16x3 stack.  It is
        # set in the device's router table, a verdict table as shipped, so the call takes the router entry point's
        # cache (_K2_ENTRY_CACHE) as with a shipped verdict; the plain dicts _force_router_rule binds are read on every
        # call, which leaves that cache unused
        arch = det_backend.arch_for(device)
        sm_count = det_backend.device_facts(device).sm_count
        table, _ = cake_backend.router_rules(arch, sm_count)
        if table is cake_backend.ROUTER_RULES:
            # a verdict of ROUTER_RULES applies on this device whatever its SM count
            monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
        key = (arch, "nn_t", int(X.shape[1]), int(G.shape[1]))
        monkeypatch.setitem(table, key, "pairs")
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()

        def call(o):
            return cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, o)

        def check(got, want):
            assert bool(torch.isfinite(got).all())
            assert _rel_fro(got, G.double().t() @ X.double()) <= ROUTER_REL_FRO_MAX

    else:
        out = torch.empty(T, W.shape[0], device=device)

        def call(o):  # no forward verdict: the deterministic program and its partials
            return cake_backend.cake_dense_gemm_nondet_train_router_forward(X, W, o)

        def check(got, want):
            assert torch.equal(got, want)

    return call, out, check


def _graph_safety_calls(case, device, monkeypatch):
    """``[small, large]``: two convenience calls of ``case`` (:data:`GRAPH_SAFETY_CASES`), each ``(call, out, check)``
    -- ``call(o)`` makes the call into ``o``, ``check(got, want)`` asserts a replay's ``got`` against the eager result
    ``want`` -- with the verdicts they need forced through ``monkeypatch``.  The large call needs more of the arena
    kinds the small one binds."""
    if case == "k1_wgrad":
        return [
            _graph_safety_k1_wgrad(row, 4097, device, SEED + 110 + i, monkeypatch)
            for i, row in enumerate(("indexer_hw", "kv_a"))
        ]
    if case == "k1_forward":
        N, K, tokens = GRAPH_SAFETY_K1_FORWARD
        return [
            _graph_safety_k1_forward(T, N, K, device, SEED + 112 + i)
            for i, T in enumerate(tokens)
        ]
    op = case.split("_")[1]
    return [
        _graph_safety_router(op, T, device, SEED + 114 + i, monkeypatch)
        for i, T in enumerate((1024, 16384) if op == "wgrad" else (4096, 16384))
    ]


def _graph_safety_bound(case, stream):
    """Asserts that the cached launches of ``stream`` bind the arena kinds of ``case``."""
    key = stream.cuda_stream
    if case == "k1_wgrad":
        entries = [e for k, e in cake_backend._K1_ATOMIC_CACHE.items() if k[1] == key]
        assert entries and all(e.plan.splits == 2 and e.k1 for e in entries)
    elif case == "k1_forward":
        entries = [e for k, e in cake_backend._K1_DET_CACHE.items() if k[1] == key]
        assert entries and all(e.k1 is not None for e in entries)
        # stream-K partial slabs: the launch writes and reads k1_det_ws (a plan without them binds the 16-element floor
        # and only adds 0 to dummy counters)
        assert all(e.plan.ws_f32_elems > 0 for e in entries), [
            (e.plan.template, e.plan.sk_units) for e in entries
        ]
    else:
        entries = [e for k, e in cake_backend._K2_ENTRY_CACHE.items() if k[1] == key]
        assert entries and all(e.launch is not None for e in entries)
        if case == "k2_wgrad":
            assert all(e.atomic and e.launch.plan.simt is None for e in entries)
        else:
            assert all(e.launch.k2.partials_shape is not None for e in entries)


def _graph_safety_scenario(case: str, mode: str) -> None:
    """A convenience call of ``case`` captured into a CUDA graph on a stream ``s`` (after a warm-up call on ``s``, so
    the capture plans and allocates nothing), then ``mode``: ``"grow"`` -- the larger call on ``s`` grows the arena,
    and an eager call of the captured key binds the grown buffers (the replaced ones lose their last binding);
    ``"evict"`` -- with a cache bound of 1, calls on another stream evict every entry of ``s`` (its arena loses its
    last cached user); under ``"grow"`` at least one of the case's kinds must have a new live buffer.  Then tensors of
    the sizes of the arena's buffers at capture, filled with :data:`GRAPH_SAFETY_FILL`, are allocated on ``s`` until one
    takes such a buffer's address or the allocator has no cached block left (the caching allocator hands a freed block
    out again on the stream it was freed on).  The replay must give the eager result and leave every filler
    untouched."""
    device, _ = _atomic_device_or_skip()
    index = torch.cuda.current_device() if device.index is None else device.index
    with pytest.MonkeyPatch.context() as mp:
        small, large = _graph_safety_calls(case, device, mp)
        call, out, check = small
        s, s2 = torch.cuda.Stream(device), torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            want = call(out).clone()
        s.synchronize()
        _graph_safety_bound(case, s)
        key = (index, s.cuda_stream)
        # no reference to the arena or its buffers is kept here (it would keep them alive)
        captured = [
            (kind, b.data_ptr(), b.numel() * b.element_size())
            for kind, b in cake_backend._ATOMIC_ARENAS[key].buffers.items()
        ]
        assert GRAPH_SAFETY_KINDS[case] <= {kind for kind, _, _ in captured}, captured
        arena_id = id(cake_backend._ATOMIC_ARENAS[key])
        print("GRAPH_SAFETY_CAPTURED", case, mode, captured, flush=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            call(out)
        if mode == "grow":
            with torch.cuda.stream(s):
                large[0](large[1])
                call(out)
        else:
            mp.setattr(det_backend, "CONVENIENCE_CACHE_SIZE", 1)
            s2.wait_stream(s)
            with torch.cuda.stream(s2):
                for c, o, _ in (small, large):
                    c(torch.empty_like(o))
            caches = (
                cake_backend._K1_ATOMIC_CACHE,
                cake_backend._K1_DET_CACHE,
                cake_backend._K2_ATOMIC_CACHE,
                cake_backend._K2_SPLITK_CACHE,
                cake_backend._K2_ENTRY_CACHE,
            )
            assert not any(k[1] == s.cuda_stream for c in caches for k in c)
        torch.cuda.synchronize()
        gc.collect()
        live = cake_backend._ATOMIC_ARENAS.get(key)
        # the captured kinds whose live buffer is no longer the one the graph was captured with
        moved = sorted(
            kind
            for kind, ptr, _ in captured
            if live is not None
            and kind in GRAPH_SAFETY_KINDS[case]
            and live.buffers[kind].data_ptr() != ptr
        )
        print(
            "GRAPH_SAFETY_ARENA",
            "kept" if id(live) == arena_id else "dropped",
            {} if live is None else {k: b.nbytes for k, b in live.buffers.items()},
            "moved",
            moved,
            flush=True,
        )
        live = None
        if mode == "grow":
            # the larger call replaced a buffer the captured launch is bound to
            assert moved, captured
        fillers, landed = [], []
        with torch.cuda.stream(s):
            for kind, ptr, nbytes in captured:
                for _ in range(64):
                    reserved = torch.cuda.memory_reserved(device)
                    filler = torch.empty(nbytes, dtype=torch.uint8, device=device)
                    filler.fill_(GRAPH_SAFETY_FILL)
                    fillers.append(filler)
                    if filler.data_ptr() == ptr:
                        landed.append(kind)
                        break
                    if torch.cuda.memory_reserved(device) != reserved:
                        break
            out.fill_(float("nan"))
            graph.replay()
        s.synchronize()
        print("GRAPH_SAFETY_FILLERS", len(fillers), "landed", landed, flush=True)
        untouched = all(bool((f == GRAPH_SAFETY_FILL).all()) for f in fillers)
        try:
            check(out, want)
            replay = "ok"
        except AssertionError as error:
            replay = f"wrong ({error!r})"[:300]
        print(
            "GRAPH_SAFETY_RESULT replay",
            replay,
            "fillers_untouched",
            untouched,
            flush=True,
        )
        assert replay == "ok" and untouched, (replay, untouched, landed)
        if case.startswith("k1"):
            assert _arena_tickets_zero()


@pytest.mark.parametrize("mode", ["grow", "evict"])
@pytest.mark.parametrize("case", GRAPH_SAFETY_CASES)
def test_gpu_captured_convenience_call_keeps_its_workspace(case, mode):
    """A convenience-path call captured into a CUDA graph stays valid until ``cake_dense_gemm_nondet_train_convenience_cache_clear``: after the
    arena grows and the captured key is bound to the grown buffers, and after cache eviction drops every entry of the
    capturing stream, its replay gives the eager result and writes nothing into tensors allocated since
    (:func:`_graph_safety_scenario`, run in a child process with a timeout: a replay on freed workspace can fault or
    spin).  The programs are built here first, so the timeout covers the scenario only."""
    device, _ = _atomic_device_or_skip()
    with pytest.MonkeyPatch.context() as mp:
        for call, out, _ in _graph_safety_calls(case, device, mp):
            call(out)
        torch.cuda.synchronize()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    root = str(Path(__file__).resolve().parents[2])
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('graph_safety', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module._graph_safety_scenario(sys.argv[2], sys.argv[3])\n"
        "print('GRAPH_SAFETY_OK')\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, *filter(None, sys.path)]))
    try:
        r = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).resolve()), case, mode],
            capture_output=True,
            text=True,
            timeout=GRAPH_SAFETY_TIMEOUT,
            cwd=root,
            env=env,
        )
    except subprocess.TimeoutExpired as expired:
        pytest.fail(
            f"{case} / {mode}: no result in {GRAPH_SAFETY_TIMEOUT} s (a replay spinning on freed workspace); "
            f"stdout {expired.stdout!r:.2000}"
        )
    assert r.returncode == 0 and "GRAPH_SAFETY_OK" in r.stdout, (
        r.returncode,
        r.stdout[-3000:],
        r.stderr[-3000:],
    )


# ---------------------------------------------------------------------------
# GPU: views, layouts and the router's FP32 contract on this package's entry points.  The tests assert the outputs,
# their layouts and the FP64 budgets; a test names a program only where it is about the weight gradient's program.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splits", [2, 3])
@pytest.mark.parametrize(
    "out_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
@pytest.mark.parametrize("row", ["thin_48", "indexer_k", "kv_a"])
def test_gpu_k1_wgrad_padded_out_runs_the_nondet_program(
    row, out_dtype, splits, k1_rule
):
    """A weight gradient into a padded ``out`` (``storage[:, :K]``, row stride ``K + 64``) at a ragged token count:
    this package's split-K program runs -- prepared at an explicit split count in both orientations of the caller's
    views (the transposed store of ``X.T @ G`` and the swapped launch of ``G.T @ X``), and through the convenience
    entry point with a verdict for the row -- by kernel name; every launch into a NaN-poisoned ``out`` meets the FP64
    budget, the padding keeps its values and the tickets are zero after."""
    device, arch = _atomic_device_or_skip()
    N, K = K1_ROWS[row]
    T = 16231
    g = torch.Generator(device=device).manual_seed(SEED + 140 + N + splits)
    G, X = _bf16((T, N), device, g), _bf16((T, K), device, g)
    storage = torch.full((N, K + 64), 7.0, device=device, dtype=out_dtype)
    out = storage[:, :K]
    ref = G.double().t() @ X.double()
    block_n = cake_backend.k1_block_n(N)
    _require_template(
        arch, cake_backend.atomic_k1_template(block_n, out_dtype == torch.float32)
    )

    def check(launch, what):
        for i in range(3):
            out.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
            _k1_check(out, ref, f"{what} {i}")
            assert bool((storage[:, K:] == 7.0).all()), what

    for A, B, transposed in ((X.t(), G, True), (G.t(), X, False)):
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            A, B, out, transposed_out=transposed, splits=splits, block_n=block_n
        )
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
        assert prepared.splits == splits
        _assert_ran(prepared.launch, prepared.module_name)
        check(prepared.launch, f"prepared transposed_out={transposed}")
        assert prepared.counters_are_zero()
    template = k1_rule(G, X, out, splits)
    module_name = cake_jit.select_module(arch, template)

    def call():
        return cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)

    _assert_ran(call, module_name)
    check(call, "convenience")
    assert _arena_tickets_zero()


def _sliced_out(rows, cols, dtype, device):
    """``(buffer, out)``: ``out [rows, cols]`` inside a NaN-filled ``[rows + 3, cols + 24]`` buffer, at row 3 and
    column 8 (a nonzero storage offset, a padded row stride, 16-byte aligned)."""
    buf = torch.full((rows + 3, cols + 24), float("nan"), device=device, dtype=dtype)
    return buf, buf[3:, 8 : 8 + cols]


def _outside_untouched(buf, cols) -> bool:
    """Whether every element of ``buf`` outside its ``out`` view (:func:`_sliced_out`) is still NaN."""
    mask = torch.ones(buf.shape, dtype=torch.bool, device=buf.device)
    mask[3:, 8 : 8 + cols] = False
    return bool(torch.isnan(buf[mask].float()).all())


@pytest.mark.parametrize("T", [301, 4097])
def test_gpu_k1_strided_views_and_storage_offsets(T, k1_rule):
    """The deterministic package's strided-view case on this package's entry points: ``X`` and ``G`` column slices of
    wider buffers at a 16-byte offset, ``W`` a row slice of a taller weight (a nonzero storage offset), every ``out`` a
    slice of a larger NaN-filled buffer with a padded row stride and a nonzero storage offset.  The forward (its entry
    point and the GEMM entry point), the input gradient (BF16 and FP32 out) and the weight gradient (BF16 and FP32 out)
    meet the FP64 budget on the convenience path and prepared, and leave the rest of each buffer untouched; the weight
    gradient with a verdict for its row runs this package's program (by kernel name) on the same views."""
    device = _det_device_or_skip()
    K, N = 704, 264
    g = torch.Generator(device=device).manual_seed(SEED + 160 + T)
    X = _bf16((T, K + 64), device, g)[:, 8 : 8 + K]
    W = _bf16((N + 16, K), device, g, K**-0.5)[16:]
    G = _bf16((T, N + 64), device, g)[:, 8 : 8 + N]
    assert X.storage_offset() and W.storage_offset() and G.storage_offset()
    bf16, f32 = torch.bfloat16, torch.float32
    fwd_ref, dgrad_ref, wgrad_ref = (
        X.double() @ W.double().t(),
        G.double() @ W.double(),
        G.double().t() @ X.double(),
    )
    # label -> (out rows, cols, dtype, the entry point's call, the prepared launch's views, FP64 reference)
    # fmt: off
    cases = {
        "fwd": (T, N, bf16, lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, o), (X, W.t()), fwd_ref),
        "fwd_gemm": (T, N, bf16, lambda o: cake_backend.cake_dense_gemm_nondet_train_projection(X, W.t(), o), (X, W.t()), fwd_ref),
        "dgrad": (T, K, bf16, lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, o), (G, W), dgrad_ref),
        "dgrad_f32": (T, K, f32, lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, o), (G, W), dgrad_ref),
        "wgrad": (N, K, bf16, lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, o), None, wgrad_ref),
        "wgrad_f32": (N, K, f32, lambda o: cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, o), None, wgrad_ref),
    }
    # fmt: on
    for label, (rows, cols, dtype, call, views, ref) in cases.items():
        buf, out = _sliced_out(rows, cols, dtype, device)
        assert out.storage_offset() and out.stride(0) == cols + 24
        if views is None:
            prepared = (
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                    G, X, out
                )
            )
        else:
            prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
                *views, out
            )
        for launch in (lambda: call(out), prepared.launch):
            for _ in range(2):
                out.fill_(float("nan"))
                assert launch() is out, label
                torch.cuda.synchronize()
                _k1_check(out, ref, label)
                assert _outside_untouched(buf, cols), label
    # the weight gradient with a verdict for its row: this package's program on the same views
    for dtype in (bf16, f32):
        buf, out = _sliced_out(N, K, dtype, device)
        module_name = cake_jit.select_module(
            det_backend.arch_for(device), k1_rule(G, X, out, 2)
        )

        def wgrad(out=out):
            return cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)

        _assert_ran(wgrad, module_name)
        for _ in range(2):
            out.fill_(float("nan"))
            assert wgrad() is out
            torch.cuda.synchronize()
            _k1_check(out, wgrad_ref, f"wgrad with a verdict {dtype}")
            assert _outside_untouched(buf, K)
        assert _arena_tickets_zero()


#: The batched MLA head projections (the deterministic package's ``MLA_ROWS``): heads, ``d_in`` (the per-head width of
#: the activations), ``d_out``, the per-head slot of the token-major activations (``d_in`` and the rope part) and the
#: weight layout (``in_out``: ``[H, d_in, d_out]``; ``out_in``: ``[H, d_out, d_in]``).
MLA_HEAD_ROWS = {
    "qabs": dict(H=64, D_in=192, D_out=512, slot=256, weight_layout="in_out"),
    "vproj": dict(H=64, D_in=512, D_out=256, slot=512, weight_layout="out_in"),
}


def _mla_launches(A, B, out, ref, what, untouched=None) -> str:
    """The generic entry point's convenience call and prepared launch of ``out = A @ B``, two of each into a
    NaN-poisoned ``out``: each returns ``out``, meets the FP64 budget and leaves ``untouched`` NaN.  An output the
    deterministic package does not offer for the views (``ValueError`` / ``NotImplementedError`` of its prepare) is
    refused by this package with the same error type; returns ``"ran"`` or ``"refused"``."""
    try:
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            A, B, out
        )
    except (ValueError, NotImplementedError) as exc:
        with pytest.raises(type(exc)):
            det_backend.cake_dense_gemm_det_train_projection_prepare(A, B, out)
        with pytest.raises(type(exc)):
            cake_backend.cake_dense_gemm_nondet_train_projection(A, B, out)
        return "refused"
    for launch in (
        prepared.launch,
        lambda: cake_backend.cake_dense_gemm_nondet_train_projection(A, B, out),
    ):
        for i in range(2):
            out.fill_(float("nan"))
            assert launch() is out, what
            torch.cuda.synchronize()
            _k1_check(out, ref, f"{what} {i}")
            if untouched is not None:
                assert bool(torch.isnan(untouched.float()).all()), what
    return "ran"


@pytest.mark.parametrize("T", [255, 4097, 16231])
@pytest.mark.parametrize("row", sorted(MLA_HEAD_ROWS))
def test_gpu_k1_mla_head_batch_layouts(row, T):
    """The batched MLA head projections at odd token counts through the generic entry point (convenience and
    prepared), every output NaN-poisoned first.  The forward writes the token-major ``[T, 64, d_out]`` buffer in place
    through its head view (``out = base.permute(1, 0, 2)``: strides ``(d_out, 64 * d_out, 1)``), so ``base.view(T, 64 *
    d_out)`` -- the value projection's ``[T, 16384]`` input of the next GEMM -- shares its data pointer and holds the
    result without a copy.  The input gradient writes the first ``d_in`` columns of each head's slot of a token-major
    ``[T, 64, slot]`` buffer (the rest untouched); the weight gradient the weight's layout, also at an explicit split
    count on this package's program.  Every output meets the FP64 budget.  FP32 gradient outputs run where the
    deterministic package offers them; elsewhere this package refuses them as that package does."""
    device = _det_device_or_skip()
    arch = det_backend.arch_for(device)
    spec = MLA_HEAD_ROWS[row]
    H, D_in, D_out, slot = spec["H"], spec["D_in"], spec["D_out"], spec["slot"]
    g = torch.Generator(device=device).manual_seed(SEED + 170 + T)
    bf16, f32 = torch.bfloat16, torch.float32
    act = _bf16((T, H, slot), device, g)[..., :D_in]
    if spec["weight_layout"] == "in_out":
        weight = _bf16((H, D_in, D_out), device, g, D_in**-0.5)
        w_in_out = weight
    else:
        weight = _bf16((H, D_out, D_in), device, g, D_in**-0.5)
        w_in_out = weight.transpose(1, 2)
    d_out = _bf16((T, H, D_out), device, g)
    outcomes = {}
    # forward: the token-major output, written in place through the head view
    base = torch.full((T, H, D_out), float("nan"), device=device, dtype=bf16)
    out = base.permute(1, 0, 2)
    assert out.stride() == (D_out, H * D_out, 1)
    A, B = act.permute(1, 0, 2), w_in_out
    outcomes["fwd"] = _mla_launches(
        A, B, out, A.double() @ B.double(), f"{row} fwd T={T}"
    )
    flat = base.view(T, H * D_out)
    assert flat.data_ptr() == base.data_ptr() == out.data_ptr()
    assert base.stride() == (H * D_out, D_out, 1)
    assert torch.equal(flat.view(T, H, D_out).permute(1, 0, 2), out)
    # input gradient: the first d_in columns of each head's slot (BF16 and FP32 out)
    A, B = d_out.permute(1, 0, 2), w_in_out.transpose(1, 2)
    ref = A.double() @ B.double()
    for dtype in (bf16, f32):
        buf = torch.full((T, H, slot), float("nan"), device=device, dtype=dtype)
        dx = buf[..., :D_in].permute(1, 0, 2)
        rest = buf[..., D_in:] if slot > D_in else None
        outcomes[f"dgrad_{dtype}"] = _mla_launches(
            A, B, dx, ref, f"{row} dgrad {dtype} T={T}", rest
        )
    del ref
    # weight gradient: the weight's layout (BF16 and FP32 out); at an explicit split count this package's program
    if spec["weight_layout"] == "in_out":
        A, B = act.permute(1, 2, 0), d_out.permute(1, 0, 2)
    else:
        A, B = d_out.permute(1, 2, 0), act.permute(1, 0, 2)
    ref = A.double() @ B.double()
    for dtype in (bf16, f32):
        dw = torch.full(weight.shape, float("nan"), device=device, dtype=dtype)
        outcomes[f"wgrad_{dtype}"] = _mla_launches(
            A, B, dw, ref, f"{row} wgrad {dtype} T={T}"
        )
        template = cake_backend.atomic_k1_template(256, dtype == f32)
        if template not in cake_jit.KERNELS.get(arch, {}):
            outcomes[f"wgrad_{dtype}_splits2"] = f"{template} not registered"
            continue
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            A, B, dw, splits=2, block_n=256
        )
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProjectionLaunch)
        assert (prepared.plan.L, prepared.splits) == (H, 2)
        _assert_ran(prepared.launch, prepared.module_name)
        for i in range(2):
            dw.fill_(float("nan"))
            prepared.launch()
            torch.cuda.synchronize()
            _k1_check(dw, ref, f"{row} wgrad {dtype} splits=2 {i}")
            assert prepared.counters_are_zero()
        outcomes[f"wgrad_{dtype}_splits2"] = "ran"
    print(f"MLA_HEAD_BATCH {arch} {row} T={T}: {outcomes}")
    assert all(v == "ran" for k, v in outcomes.items() if "float32" not in k), outcomes


@pytest.fixture
def tf32_settings():
    """Restore PyTorch's TF32 settings (``allow_tf32`` of cuBLAS and cuDNN, the float32 matmul precision) after the
    test."""
    matmul, cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    precision = torch.get_float32_matmul_precision()
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("T", [16172, 16231])
@pytest.mark.parametrize("hidden", [4096, 5120])
def test_gpu_router_ignores_the_tf32_settings(hidden, T, tf32_settings):
    """The router GEMM stays FP32 (no silent TF32 or BF16): with ``torch.backends.cuda.matmul.allow_tf32 = True`` and
    ``torch.set_float32_matmul_precision("high")`` every router call -- the forward with and without a split count,
    the input gradient, the weight gradient with and without one; the entry points from the keys planned before the
    setting and planned under it, and the prepared launches -- meets the router's FP64 budget (the forward's top-8
    experts too), and a launch that repeats bit for bit gives the bits of the default-precision run."""
    device, arch = _atomic_device_or_skip()
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    X, W, G = _router_inputs(T, device, SEED + 180 + hidden + T, K=hidden)
    tensors = {"fwd": (X, W), "dgrad": (G, W), "wgrad": (G, X)}
    gemms = {"fwd": (X, W.t()), "dgrad": (G, W), "wgrad": (G.t(), X)}
    calls = {
        "fwd": cake_backend.cake_dense_gemm_nondet_train_router_forward,
        "dgrad": cake_backend.cake_dense_gemm_nondet_train_router_dgrad,
        "wgrad": cake_backend.cake_dense_gemm_nondet_train_router_wgrad,
    }
    shapes = {"fwd": (T, 256), "dgrad": (T, hidden), "wgrad": (256, hidden)}
    refs = {op: A.double() @ B.double() for op, (A, B) in gemms.items()}
    cases = (("fwd", None), ("fwd", 2), ("dgrad", None), ("wgrad", None), ("wgrad", 2))
    stream = torch.cuda.current_stream(device)

    def run() -> dict:
        """label -> (output, whether its launch repeats bit for bit)."""
        results = {}
        for op, splits in cases:
            P, Q = tensors[op]
            kw = {} if op == "dgrad" else {"splits": splits}
            out = torch.full(shapes[op], float("nan"), device=device)
            assert calls[op](P, Q, out, **kw) is out
            torch.cuda.synchronize()
            entry = _router_entry_of(op, P, Q, out, splits, stream)
            bits = entry is not None and entry.launch is not None
            results[(op, splits, "entry")] = (out, bits and _router_entry_bits(entry))
            pout = torch.full(shapes[op], float("nan"), device=device)
            prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
                *gemms[op], pout, **({} if splits is None else {"splits": splits})
            )
            prepared.launch()
            torch.cuda.synchronize()
            varies = (
                isinstance(prepared, cake_backend.CakeDenseGemmNondetRouterLaunch)
                and not prepared.plan.reproducible
            )
            results[(op, splits, "prepared")] = (pout, not varies)
        return results

    def check(results, what):
        for (op, splits, path), (out, _) in results.items():
            label = (what, op, splits, path)
            assert bool(torch.isfinite(out).all()), label
            assert _rel_fro(out, refs[op]) <= ROUTER_REL_FRO_MAX, label
            if op == "fwd":
                assert _top8_rows_differ(out, refs[op])[0] == 0, label

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    default = run()
    check(default, "default precision")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    compared = 0
    for what in ("TF32 on, keys planned before", "TF32 on, planned under it"):
        if what.endswith("under it"):
            cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        results = run()
        check(results, what)
        for label, (out, bits) in results.items():
            want, want_bits = default[label]
            if bits and want_bits:
                assert torch.equal(out, want), (what, label)
                compared += 1
    print(
        f"ROUTER_TF32 {arch} H{hidden} T{T}: {compared} launches compared bit for bit"
    )
    assert compared


# ---------------------------------------------------------------------------
# Measured routing of the repeatable forms: named deterministic programs of the input gradient, the forward and the
# weight gradient; the weight gradient without an arrival-order sum of three or more partials; the router weight
# gradient on the deterministic program at a per-row split count
# ---------------------------------------------------------------------------

#: The parts of :data:`K1_DGRAD_RULES_BY_SM` and the L2 size the CPU planner is given for each.
K1_DGRAD_PARTS = {
    ("sm_100a", 148): 126 << 20,
    ("sm_100a", 152): GB_L2_BYTES,
    ("sm_103a", 152): GB_L2_BYTES,
    ("sm_107a", 212): 126 << 20,
}
#: The programs of :data:`K1_DGRAD_RULES_BY_SM` and the output dtype each runs.
K1_DGRAD_PROGRAMS = {
    "kn_n128_f32_tma1_pd2_c1": "f32",
    "kn_n128_f32_tma2_s4_c1": "f32",
    "kn_n256_f32_v8_ef_pro7": "f32",
    "kn_n256_tma2_pd2_lw_pro15": "bf16",
    "kn_n256_tma1_w8": "bf16",
    "kn_n256_tma1_w8r": "bf16",
    "kn_n256_tma1_w8rs": "bf16",
}
#: The rows per part of :data:`K1_DGRAD_RULES_BY_SM`.
K1_DGRAD_ROW_COUNTS = {
    ("sm_100a", 148): 74,
    ("sm_100a", 152): 66,
    ("sm_103a", 152): 67,
    ("sm_107a", 212): 9,
}
#: The measured token counts of the input-gradient and named forward rows.
K1_SCREEN_TOKENS = (16172, 16231, 16384)


def _plan_of_knobs(A, B, out, arch, sms, knobs, transposed_out=False):
    """The deterministic planner's plan of ``(A, B, out)`` on a part of ``sms`` SMs (CPU tensors, nothing launched)."""
    l2 = K1_DGRAD_PARTS.get((arch, sms), GB_L2_BYTES)
    return det_backend.cake_dense_gemm_det_train_projection_plan(
        A,
        B,
        out,
        sm_count=sms,
        l2_bytes=l2,
        arch=arch,
        transposed_out=transposed_out,
        **dict(knobs),
    )[0]


def test_k1_dgrad_rules_by_sm():
    """:data:`K1_DGRAD_RULES_BY_SM` holds input-gradient rows of at most 1024 out features at the measured token
    counts only, on the measured parts, each naming a program of the deterministic package
    for its output dtype; the deterministic planner turns the knobs of each row (:func:`k1_program_knobs`) into exactly
    that program, with whole output tiles (no stream-K units).  :func:`k1_dgrad_knobs` keys exactly: another part,
    token count, row or output dtype gets no knobs (the deterministic plan)."""
    table = cake_backend.K1_DGRAD_RULES_BY_SM
    assert set(table) == set(K1_DGRAD_PARTS)
    assert {part: len(rows) for part, rows in table.items()} == K1_DGRAD_ROW_COUNTS
    bf16 = torch.bfloat16
    for (arch, sms), rows in table.items():
        for (K, N, T, dt), program in rows.items():
            what = (arch, sms, K, N, T, dt, program)
            assert N <= 1024 < K and T in K1_SCREEN_TOKENS, what
            assert K1_DGRAD_PROGRAMS[program] == dt, what
            assert program != "kn_n256_f32_v8_ef_pro7" or N == 1024, what
            f32 = dt == "f32"
            knobs = cake_backend.k1_dgrad_knobs(
                arch=arch, sm_count=sms, K=K, N=N, T=T, out_f32=f32
            )
            assert knobs == cake_backend.k1_program_knobs(program), what
            G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
            out = torch.empty(T, K, dtype=torch.float32 if f32 else bf16)
            plan = _plan_of_knobs(G, W, out, arch, sms, knobs)
            assert plan.template == det_backend._TEMPLATE_PREFIX + program, (
                what,
                plan.template,
            )
            assert int(plan.sk_units) == 0, what
            # exact keys: the other output dtype, the token counts next to it, other parts (the promotion of a
            # range of K1_DGRAD_PROMO_RANGES_BY_SM where one holds the key)
            other = dict(arch=arch, K=K, N=N)

            def promo_of(t, out_f32, sms=sms):
                promo = cake_backend.k1_dgrad_promo(
                    sm_count=sms, T=t, out_f32=out_f32, **other
                )
                return (("promo", promo),) if promo else ()

            if (K, N, T, "bf16" if f32 else "f32") not in rows:
                assert cake_backend.k1_dgrad_knobs(
                    sm_count=sms, T=T, out_f32=not f32, **other
                ) == promo_of(T, not f32), what
            for t in (T - 1, T + 1):
                if (K, N, t, dt) not in rows:
                    assert cake_backend.k1_dgrad_knobs(
                        sm_count=sms, T=t, out_f32=f32, **other
                    ) == promo_of(t, f32), what
            for s in (sms - 1, sms + 1, 160, 216):
                assert (
                    cake_backend.k1_dgrad_knobs(sm_count=s, T=T, out_f32=f32, **other)
                    == ()
                ), what
    assert (
        cake_backend.k1_dgrad_knobs(
            arch="sm_107a", sm_count=212, K=4096, N=32, T=16231, out_f32=True
        )
        == ()
    )
    with pytest.raises(ValueError):
        cake_backend.k1_program_knobs("kn_n128_f32_tma2_s4_c1_box64")


def test_k1_program_knobs_are_planner_keywords(monkeypatch):
    """Every knob set this package hands the deterministic package's planner names keywords of
    ``cake_dense_gemm_det_train_projection_plan``: the knobs of every deterministic program its tables name
    (:data:`K1_DGRAD_RULES_BY_SM`, :data:`K1_WGRAD_DET_RULES_BY_SM`, the program entries of
    :data:`K1_FORWARD_RULES_BY_SM` and :data:`K1_DGRAD_NVJET_DET_PROGRAM`, through :func:`k1_program_knobs`) and the
    knob entries of :data:`K1_FORWARD_RULES_BY_SM`.  A knob that package's symbol inverse adds but its planner does not
    take fails here, not at a call on the GPU.  :func:`k1_program_knobs` drops a serpentine K walk (``kser``) of 0 and
    refuses a ``_ks<P>`` program, which that package plans from its own rules only."""
    keywords = set(
        inspect.signature(
            det_backend.cake_dense_gemm_det_train_projection_plan
        ).parameters
    )
    programs = {cake_backend.K1_DGRAD_NVJET_DET_PROGRAM}
    for table in (
        cake_backend.K1_DGRAD_RULES_BY_SM,
        cake_backend.K1_WGRAD_DET_RULES_BY_SM,
    ):
        programs |= {program for rows in table.values() for program in rows.values()}
    knob_sets = []
    for rows in cake_backend.K1_FORWARD_RULES_BY_SM.values():
        for value in rows.values():
            if isinstance(value, str):
                programs.add(value)
            else:
                knob_sets.append(("forward knobs", dict(value)))
    knob_sets += [
        (program, dict(cake_backend.k1_program_knobs(program)))
        for program in sorted(programs)
    ]
    assert len(knob_sets) > len(programs) > 1
    for what, knobs in knob_sets:
        assert knobs and set(knobs) <= keywords, (what, sorted(set(knobs) - keywords))
    # the serpentine K walk of the symbol inverse: dropped at 0, a ``_ks<P>`` program names no planner knobs
    inverse = det_backend._template_knobs
    program = cake_backend.K1_DGRAD_NVJET_DET_PROGRAM
    plain = cake_backend.k1_program_knobs(program)
    monkeypatch.setattr(
        det_backend,
        "_template_knobs",
        lambda name, *a, **k: {**inverse(name, *a, **k), "kser": 0},
    )
    assert cake_backend.k1_program_knobs(program) == plain
    monkeypatch.setattr(
        det_backend,
        "_template_knobs",
        lambda name, *a, **k: {**inverse(name, *a, **k), "kser": 76},
    )
    with pytest.raises(ValueError):
        cake_backend.k1_program_knobs(program)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_k1_dgrad_rows_bind_their_plan(monkeypatch, arch):
    """``cake_dense_gemm_nondet_train_projection_dgrad()`` binds, for a row of :data:`K1_DGRAD_RULES_BY_SM` on its
    part (both operands with a unit stride on their last axis), the program of the row's knobs, and the deterministic
    plan for a token count, output dtype or SM count outside the table; an operand without a unit stride on its last
    axis gets no knobs.  The rows of :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` and :data:`K1_DGRAD_NVJET_RULES_BY_SM`,
    which come first, are taken out (``test_k1_program_rows_bind_their_program`` and
    ``test_k1_dgrad_nvjet_rows_bind_the_nvjet_program`` hold them)."""
    _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    monkeypatch.setattr(cake_backend, "K1_DGRAD_PROGRAM_RULES_BY_SM", {})
    monkeypatch.setattr(cake_backend, "K1_DGRAD_NVJET_RULES_BY_SM", {})
    bf16 = torch.bfloat16
    rows = cake_backend.K1_DGRAD_RULES_BY_SM[(arch, 152)]
    picked = sorted(rows.items())[:: max(1, len(rows) // 6)]

    def bound_plan(G, W, out):
        cake_backend._K1_DET_CACHE.clear()
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out) is out
        )
        ((key, entry),) = cake_backend._K1_DET_CACHE.items()
        assert key[2][0] == ("dgrad",) and entry.k1 is not None
        return entry.plan

    for (K, N, T, dt), program in picked:
        f32 = dt == "f32"
        G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out = torch.empty(T, K, dtype=torch.float32 if f32 else bf16)
        knobs = cake_backend.k1_program_knobs(program)
        assert cake_backend._k1_dgrad_knobs_of(G, W, out) == knobs
        plan = bound_plan(G, W, out)
        want = det_backend._k1_plan(G, W, out, transposed_out=False, knobs=dict(knobs))[
            0
        ]
        assert plan == want and plan.template.endswith("_" + program), (K, N, T, dt)
        for t in (T - 1, T + 1):
            if (K, N, t, dt) in rows:
                continue
            Gt = torch.empty(t, N, dtype=bf16)
            outt = torch.empty(t, K, dtype=out.dtype)
            assert cake_backend._k1_dgrad_knobs_of(Gt, W, outt) == ()
            plain = det_backend._k1_plan(Gt, W, outt, transposed_out=False, knobs={})[0]
            assert bound_plan(Gt, W, outt) == plain
        # an operand without a unit stride on its last axis: no knobs
        Wmn = torch.empty(K, N, dtype=bf16).t()
        assert cake_backend._k1_dgrad_knobs_of(G, Wmn, out) == ()
    # another SM count of the architecture: no knobs
    (K, N, T, dt), _ = picked[0]
    G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
    out = torch.empty(T, K, dtype=torch.float32 if dt == "f32" else bf16)
    for sms in (150, 151, 153, 160):
        facts = dataclasses.replace(det_backend.device_facts(), sm_count=sms)
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        assert cake_backend._k1_dgrad_knobs_of(G, W, out) == (), sms


#: The rows of :data:`K1_DGRAD_PROMO_RANGES_BY_SM` (148-SM ``sm_100a``, BF16 out, 16172 to 16384 tokens).
K1_DGRAD_PROMO_ROWS = (
    (4096, 512),
    (4096, 1024),
    (5120, 512),
    (5120, 1024),
    (6144, 576),
)


def test_k1_dgrad_promo_ranges_bind_the_promoted_plan(monkeypatch):
    """:data:`K1_DGRAD_PROMO_RANGES_BY_SM` holds the BF16 input-gradient rows of :data:`K1_DGRAD_PROMO_ROWS` on
    148-SM ``sm_100a`` parts from 16172 to 16384 tokens with the TMA L2 promotion of 256 bytes.  :func:`k1_dgrad_knobs`
    gives that ``promo`` inside the range only (BF16 out, exactly 148 SMs), and
    ``cake_dense_gemm_nondet_train_projection_dgrad()`` binds the deterministic plan of the row with it -- every field
    of that plan, which promotes none, but the promotion, so the same program, grid and bits -- and the deterministic
    plan outside the range.  The generic prepare routes a row of the range to the input-gradient entry point and names
    the table and no template (a launch parameter, not a program); an operand without a unit stride on its last axis
    and another SM count get no knobs."""
    table = cake_backend.K1_DGRAD_PROMO_RANGES_BY_SM
    assert table == {
        ("sm_100a", 148): {
            (K, N, "bf16"): ((16172, 16384, "l2_256b"),) for K, N in K1_DGRAD_PROMO_ROWS
        }
    }
    _cpu_k1_launches(monkeypatch, "sm_100a")
    facts = dataclasses.replace(det_backend.device_facts(), sm_count=148)
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    rules = cake_backend.K1_DGRAD_RULES_BY_SM[("sm_100a", 148)]

    def bound_plan(G, W, out):
        cake_backend._K1_DET_CACHE.clear()
        assert (
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out) is out
        )
        ((key, entry),) = cake_backend._K1_DET_CACHE.items()
        assert key[2][0] == ("dgrad",) and entry.k1 is not None
        return entry.plan

    for K, N in K1_DGRAD_PROMO_ROWS:
        W = torch.empty(N, K, dtype=bf16)
        for T in (16171, 16172, 16231, 16300, 16384, 16385):
            inside = 16172 <= T <= 16384
            G = torch.empty(T, N, dtype=bf16)
            for f32 in (False, True):
                what = (K, N, T, f32)
                out = torch.empty(T, K, dtype=torch.float32 if f32 else bf16)
                program = rules.get((K, N, T, "f32" if f32 else "bf16"))
                if program:
                    want = cake_backend.k1_program_knobs(program)
                else:
                    want = (("promo", "l2_256b"),) if inside and not f32 else ()
                knobs = cake_backend.k1_dgrad_knobs(
                    arch="sm_100a", sm_count=148, K=K, N=N, T=T, out_f32=f32
                )
                assert knobs == want, what
                assert cake_backend._k1_dgrad_knobs_of(G, W, out) == want, what
                for arch, sms in (("sm_100a", 152), ("sm_103a", 152), ("sm_107a", 212)):
                    other = cake_backend.k1_dgrad_knobs(
                        arch=arch, sm_count=sms, K=K, N=N, T=T, out_f32=f32
                    )
                    assert "promo" not in dict(other), (what, arch, sms)
                if f32:
                    continue  # FP32 out: the other input-gradient tables of the part
                plain = det_backend._k1_plan(G, W, out, transposed_out=False, knobs={})[
                    0
                ]
                plan = bound_plan(G, W, out)
                if not inside:
                    assert plan == plain, what
                    continue
                assert plain.promo == "none" and plan.promo == "l2_256b", what
                assert plan == dataclasses.replace(plain, promo="l2_256b"), what
                routed = cake_backend._k1_table_call(G, W, out, False)
                assert routed is not None and routed[:2] == (
                    "K1_DGRAD_PROMO_RANGES_BY_SM",
                    (K, N, T, "bf16"),
                ), what
                prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
                    G, W, out
                )
                assert isinstance(
                    prepared, cake_backend.CakeDenseGemmNondetProgramLaunch
                )
                assert (prepared.table, prepared.template, prepared.own) == (
                    "K1_DGRAD_PROMO_RANGES_BY_SM",
                    None,
                    False,
                ), what
        # an operand without a unit stride on its last axis: no knobs
        G, out = torch.empty(16231, N, dtype=bf16), torch.empty(16231, K, dtype=bf16)
        Wmn = torch.empty(K, N, dtype=bf16).t()
        assert cake_backend._k1_dgrad_knobs_of(G, Wmn, out) == ()
    # another SM count of the architecture: no knobs
    K, N = K1_DGRAD_PROMO_ROWS[0]
    G, W = torch.empty(16231, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
    out = torch.empty(16231, K, dtype=bf16)
    for sms in (144, 147, 149, 152):
        facts = dataclasses.replace(det_backend.device_facts(), sm_count=sms)
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        assert cake_backend._k1_dgrad_knobs_of(G, W, out) == (), sms


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_k1_dgrad_nvjet_rows_bind_the_nvjet_program(monkeypatch, arch):
    """``cake_dense_gemm_nondet_train_projection_dgrad()`` binds, for a row of :data:`K1_DGRAD_NVJET_RULES_BY_SM` on its
    part, this package's nvjet program on the deterministic package's plan of :data:`K1_DGRAD_NVJET_DET_PROGRAM`: the
    cached entry holds that plan and names the program, and the bound arguments are the plan's arguments in the
    program's argument order -- the argument plan and the cluster of the deterministic program.
    Another token count, FP32 out, another SM count, an operand without a unit stride on its last axis and a measured
    row the part leaves to the other tables (``4096 -> 32`` at 2048 tokens) keep the other tables (no program)."""
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    rows = cake_backend.K1_DGRAD_NVJET_RULES_BY_SM[(arch, 152)]
    knobs = dict(cake_backend.k1_program_knobs(cake_backend.K1_DGRAD_NVJET_DET_PROGRAM))
    picked = sorted(rows.items())[:: max(1, len(rows) // 6)]
    for (K, N, T, dt), template in picked:
        what = (K, N, T, dt)
        module = cake_jit.KERNELS[arch][template]
        G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out = torch.empty(T, K, dtype=bf16)
        assert cake_backend._k1_dgrad_nvjet_of(G, W, out) == module, what
        cake_backend._K1_DET_CACHE.clear()
        entry_fn.calls.clear()
        for _ in range(2):
            got = cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out)
            assert got is out, what
        assert len(entry_fn.calls) == 2, what
        ((key, entry),) = cake_backend._K1_DET_CACHE.items()
        assert key[2][0] == ("dgrad",) and entry.k1 is not None, what
        want = det_backend._k1_plan(G, W, out, transposed_out=False, knobs=knobs)
        plan, *_, det_module = want
        assert entry.plan == plan and entry.module_name == module, what
        assert plan.template == (
            det_backend._TEMPLATE_PREFIX + cake_backend.K1_DGRAD_NVJET_DET_PROGRAM
        )
        # the same argument plan and cluster as the deterministic program
        record = cake_jit.MODULES[module]
        det_record = det_backend.MODULES[det_module]
        arg_plan = cake_jit.arg_plan_of(record)
        assert arg_plan == det_backend.arg_plan_of(det_record), what
        assert record["launch"]["cluster"] == det_record["launch"]["cluster"], what
        (args, _) = entry_fn.calls[-1]
        assert len(args) == len(arg_plan), what
        for t in (T - 1, T + 1):
            if (K, N, t, dt) in rows:
                continue
            Gt, outt = torch.empty(t, N, dtype=bf16), torch.empty(t, K, dtype=bf16)
            assert cake_backend._k1_dgrad_nvjet_of(Gt, W, outt) is None, (what, t)
        f32 = torch.empty(T, K, dtype=torch.float32)
        assert cake_backend._k1_dgrad_nvjet_of(G, W, f32) is None, what
        Wmn = torch.empty(K, N, dtype=bf16).t()
        assert cake_backend._k1_dgrad_nvjet_of(G, Wmn, out) is None, what
    kept = _NVJET_SM100_KEPT if arch == "sm_100a" else _NVJET_SM103_KEPT
    assert (4096, 32, 2048) in kept
    for K, N, T in kept:
        G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out = torch.empty(T, K, dtype=bf16)
        assert (K, N, T, "bf16") not in rows
        assert cake_backend._k1_dgrad_nvjet_of(G, W, out) is None, (K, N, T)
    (K, N, T, dt), _ = picked[0]
    G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
    out = torch.empty(T, K, dtype=bf16)
    for sms in (148, 150, 153, 160):
        if (arch, sms) in cake_backend.K1_DGRAD_NVJET_RULES_BY_SM:
            continue
        facts = det_backend.DeviceFacts(
            index=0, arch=arch, sm_count=sms, l2_bytes=126 << 20
        )
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        assert cake_backend._k1_dgrad_nvjet_of(G, W, out) is None, sms


def test_k1_dgrad_nvjet_sm107a_rows_bind_the_nvjet_program(monkeypatch):
    """On a 212-SM ``sm_107a`` part, where the deterministic package does not register
    :data:`K1_DGRAD_NVJET_DET_PROGRAM`, ``cake_dense_gemm_nondet_train_projection_dgrad()`` binds for a row of
    :data:`K1_DGRAD_NVJET_RULES_BY_SM` this package's nvjet program on the deterministic planner's plan of that
    program: the cached entry holds that plan and names the program, with the deterministic program's argument plan
    and cluster.  Another token count and another SM count keep the other tables (no program)."""
    arch, sms = "sm_107a", 212
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sms, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    rows = cake_backend.K1_DGRAD_NVJET_RULES_BY_SM[(arch, sms)]
    knobs = dict(cake_backend.k1_program_knobs(cake_backend.K1_DGRAD_NVJET_DET_PROGRAM))
    det_template = (
        det_backend._TEMPLATE_PREFIX + cake_backend.K1_DGRAD_NVJET_DET_PROGRAM
    )
    det_record = det_backend.MODULES[det_backend.select_module("sm_100a", det_template)]
    for (K, N, T, dt), template in sorted(rows.items())[::4]:
        what = (K, N, T, dt)
        module = cake_jit.KERNELS[arch][template]
        G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out = torch.empty(T, K, dtype=bf16)
        assert cake_backend._k1_dgrad_nvjet_of(G, W, out) == module, what
        cake_backend._K1_DET_CACHE.clear()
        entry_fn.calls.clear()
        for _ in range(2):
            assert (
                cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, out)
                is out
            ), what
        assert len(entry_fn.calls) == 2, what
        ((key, entry),) = cake_backend._K1_DET_CACHE.items()
        assert key[2][0] == ("dgrad",) and entry.k1 is not None, what
        plan = det_backend.cake_dense_gemm_det_train_projection_plan(
            G,
            W,
            out,
            sm_count=sms,
            l2_bytes=facts.l2_bytes,
            transposed_out=False,
            arch=arch,
            **knobs,
        )[0]
        assert entry.plan == plan and entry.module_name == module, what
        assert plan.template == det_template, what
        record = cake_jit.MODULES[module]
        assert cake_jit.arg_plan_of(record) == det_backend.arg_plan_of(det_record), what
        assert record["launch"]["cluster"] == det_record["launch"]["cluster"], what
        assert len(entry_fn.calls[-1][0]) == len(cake_jit.arg_plan_of(record)), what
        for t in (T - 1, T + 1):
            if (K, N, t, dt) not in rows:
                Gt, outt = torch.empty(t, N, dtype=bf16), torch.empty(t, K, dtype=bf16)
                assert cake_backend._k1_dgrad_nvjet_of(Gt, W, outt) is None, (what, t)
    (K, N, T, _), _ = min(rows.items())
    G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
    out = torch.empty(T, K, dtype=bf16)
    for other in (208, 211, 213, 216):
        facts = det_backend.DeviceFacts(
            index=0, arch=arch, sm_count=other, l2_bytes=126 << 20
        )
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        assert cake_backend._k1_dgrad_nvjet_of(G, W, out) is None, other


#: The program the rows of :data:`K1_FORWARD_RULES_BY_SM` that name one run per part, and the deterministic plan of
#: those rows there: ``(arch, sm_count) -> (program, deterministic plan)``.
K1_FORWARD_NAMED = {
    ("sm_100a", 148): {"kk_n256_ht_cg_pro15": "kk_n256_pro7"},
    ("sm_100a", 152): {"kk_n256_ht_cg_pro15": "kk_n256_pro7"},
    ("sm_103a", 152): {"kk_n256_ht_cg_pro15": "kk_n256_pro7"},
    ("sm_107a", 212): {
        "kk_n128_hen_pro15": "kk_n128_hen_pro7",
        "kk_n192_m256_hen_s5_pro15": "kk_n256_ht_cg_pro15",
    },
}


def test_k1_forward_named_rows_plan_their_program():
    """The rows of :data:`K1_FORWARD_RULES_BY_SM` that name a program (:data:`K1_FORWARD_NAMED`, at the measured
    token counts of ``4096 -> 512`` and ``5120 -> 512``, and on 212-SM ``sm_107a`` parts of ``5120 -> 32`` / ``48`` /
    ``64`` and ``8192 -> 128``) plan exactly that program with the knobs that name it, where the deterministic plan of
    the row is the program :data:`K1_FORWARD_NAMED` pairs with it."""
    named = 0
    bf16 = torch.bfloat16
    for (arch, sms), rows in cake_backend.K1_FORWARD_RULES_BY_SM.items():
        for (K, N, T), value in rows.items():
            if not isinstance(value, str):
                continue
            program, plain = value, K1_FORWARD_NAMED[(arch, sms)][value]
            assert T in K1_SCREEN_TOKENS, (arch, sms, K, N, T)
            assert N == 512 or (arch, program) == ("sm_107a", "kk_n128_hen_pro15")
            knobs = cake_backend.k1_forward_knobs(
                arch=arch, sm_count=sms, K=K, N=N, T=T
            )
            assert knobs == cake_backend.k1_program_knobs(value)
            X, W = torch.empty(T, K, dtype=bf16), torch.empty(N, K, dtype=bf16)
            out = torch.empty(T, N, dtype=bf16)
            own = _plan_of_knobs(X, W.t(), out, arch, sms, knobs)
            default = _plan_of_knobs(X, W.t(), out, arch, sms, ())
            assert own.template.endswith("_" + program), (arch, sms, K, N, T)
            assert default.template.endswith("_" + plain), (
                arch,
                sms,
                K,
                N,
                T,
                default.template,
            )
            named += 1
    assert named == 25


#: The B200 weight-gradient rows of :data:`K1_WGRAD_DET_RULES_BY_SM`: (d_in, d_out) -> program per output dtype.
K1_WGRAD_DET_ROWS = {
    (5120, 1024): {
        "bf16": "nn_n192_bz64_hne_sks_sb2_pro15",
        "f32": "nn_n192_bz64_hne_f32_v8_ef_sks_sb2_pro15",
    },
}


def test_k1_wgrad_det_rules_by_sm():
    """:data:`K1_WGRAD_DET_RULES_BY_SM` holds the 148-SM ``sm_100a`` ``5120 -> 1024`` rows at the measured token
    counts, BF16 and FP32; the deterministic planner turns their knobs, in its weight-gradient orientation, into the
    named two-range programs, and :func:`k1_wgrad_det_knobs` keys exactly.  :func:`plan_atomic_gemm` keeps these rows
    on the deterministic program."""
    table = cake_backend.K1_WGRAD_DET_RULES_BY_SM
    assert set(table) == {("sm_100a", 148)}
    rows = table[("sm_100a", 148)]
    assert rows == {
        (d_in, d_out, T, dt): program
        for (d_in, d_out), programs in K1_WGRAD_DET_ROWS.items()
        for T in K1_SCREEN_TOKENS
        for dt, program in programs.items()
    }
    bf16 = torch.bfloat16
    for (d_in, d_out, T, dt), program in rows.items():
        f32 = dt == "f32"
        kw = dict(arch="sm_100a", d_in=d_in, d_out=d_out, T=T, out_f32=f32)
        knobs = cake_backend.k1_wgrad_det_knobs(sm_count=148, **kw)
        assert knobs == cake_backend.k1_program_knobs(program)
        for sms in (147, 149, 152):
            assert cake_backend.k1_wgrad_det_knobs(sm_count=sms, **kw) == ()
        for t in (T - 1, T + 1):
            if t not in K1_SCREEN_TOKENS:
                kt = dict(kw, T=t)
                assert cake_backend.k1_wgrad_det_knobs(sm_count=148, **kt) == ()
        G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
        out = torch.empty(d_out, d_in, dtype=torch.float32 if f32 else bf16)
        A, B, transposed = det_backend.wgrad_views(G, X, arch="sm_100a", out_f32=f32)
        plan = _plan_of_knobs(
            A, B, out, "sm_100a", 148, knobs, transposed_out=transposed
        )
        assert plan.template == det_backend._TEMPLATE_PREFIX + program, (
            T,
            dt,
            plan.template,
        )
        assert (
            cake_backend.plan_atomic_gemm(
                X.t(), G, out, transposed_out=True, arch="sm_100a", sm_count=148
            )[0]
            is None
        )


#: Weight-gradient rows whose route changed, per part, at the bench token counts: ``(d_in, d_out, dtype, T) ->
#: (kind, block_n, splits)`` -- ``"dsm"`` the cluster program (also the cluster tile of a range of
#: :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`), ``"pair"`` the CTA-pair program, ``"ordered"`` / ``"odd"`` the CTA-pair
#: program of a range of :data:`K1_ORDERED_RANGES_BY_SM` / :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`, ``"sk"`` the
#: stream-K program, ``"det"`` the deterministic program (``"det"``, ``None``, ``None``).
# fmt: off
K1_WGRAD_CHANGED_ROWS = {
    ("sm_100a", 148): {
        (4096, 48, "f32", 16172): ("dsm", 64, 4),
        (4096, 48, "f32", 16231): ("dsm", 32, 2), (4096, 48, "f32", 16384): ("dsm", 32, 2),
        (4096, 64, "f32", 16172): ("dsm", 64, 4),
        (4096, 64, "f32", 16231): ("dsm", 32, 2), (4096, 64, "f32", 16384): ("dsm", 32, 2),
        (4096, 1024, "bf16", 16172): ("ordered", 256, 1), (4096, 1024, "bf16", 16231): ("ordered", 256, 1),
        (4096, 1024, "bf16", 16384): ("ordered", 256, 1),
        (5120, 1024, "bf16", 16172): ("sk", 256, None), (5120, 1024, "bf16", 16231): ("sk", 256, None),
        (5120, 1024, "bf16", 16384): ("sk", 256, None), (5120, 1024, "f32", 16172): ("sk", 256, None),
        (5120, 1024, "f32", 16231): ("sk", 256, None), (5120, 1024, "f32", 16384): ("sk", 256, None),
    },
    ("sm_100a", 152): {
        (4096, 48, "f32", 16172): ("dsm", 64, 4),
        (4096, 48, "f32", 16231): ("dsm", 32, 2), (4096, 48, "f32", 16384): ("dsm", 32, 2),
        (4096, 64, "f32", 16172): ("dsm", 64, 4),
        (4096, 64, "f32", 16231): ("dsm", 32, 2), (4096, 64, "f32", 16384): ("dsm", 32, 2),
    },
    ("sm_103a", 152): {
        (4096, 32, "f32", 16231): ("dsm", 32, 2),
        (4096, 48, "f32", 16172): ("dsm", 64, 4),
        (4096, 48, "f32", 16231): ("dsm", 32, 2), (4096, 48, "f32", 16384): ("dsm", 32, 2),
        (4096, 64, "f32", 16172): ("dsm", 64, 4),
        (4096, 64, "f32", 16231): ("dsm", 32, 2), (4096, 64, "f32", 16384): ("dsm", 32, 2),
        (5120, 32, "f32", 16384): ("det", None, None), (5120, 48, "bf16", 16231): ("det", None, None),
        (5120, 64, "f32", 16231): ("det", None, None), (5120, 64, "f32", 16384): ("det", None, None),
        (5120, 128, "f32", 16231): ("det", None, None), (5120, 128, "f32", 16384): ("det", None, None),
        (5120, 1024, "bf16", 16172): ("sk", 256, None), (5120, 1024, "f32", 16231): ("sk", 256, None),
        (5120, 1024, "f32", 16384): ("sk", 256, None),
    },
    ("sm_107a", 212): {
        (4096, 32, "f32", 16172): ("ordered", 64, 4),
        (4096, 32, "f32", 16231): ("ordered", 64, 4), (4096, 32, "f32", 16384): ("ordered", 64, 4),
        (5120, 32, "f32", 16231): ("pair", 64, 2), (5120, 32, "f32", 16384): ("pair", 64, 2),
        (5120, 48, "f32", 16384): ("pair", 64, 2),
        (4096, 48, "f32", 16172): ("dsm", 64, 4),
        (4096, 48, "f32", 16231): ("dsm", 64, 4), (4096, 48, "f32", 16384): ("dsm", 64, 4),
        (4096, 64, "f32", 16172): ("dsm", 64, 4),
        (4096, 64, "f32", 16231): ("dsm", 64, 4), (4096, 64, "f32", 16384): ("dsm", 64, 4),
        (5120, 64, "f32", 16384): ("det", None, None),
        (4096, 1024, "f32", 16231): ("ordered", 192, 1), (4096, 1024, "f32", 16384): ("ordered", 192, 1),
        (4096, 1024, "bf16", 16172): ("sk", 256, None), (4096, 1024, "bf16", 16384): ("sk", 256, None),
        (4096, 1024, "f32", 16172): ("ordered", 192, 1),
        (5120, 1024, "f32", 16172): ("ordered", 256, 1), (5120, 1024, "f32", 16231): ("ordered", 256, 1),
        (5120, 1024, "f32", 16384): ("ordered", 256, 1),
        (5120, 512, "f32", 16231): ("ordered", 128, 1), (5120, 512, "f32", 16384): ("ordered", 128, 1),
    },
}
# fmt: on


def _k1_fixed_program(arch, sms, d_in, d_out, T, f32, L=1):
    """``(block_n, splits, chains, order)`` of the CTA-pair program a row runs from a range of
    :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` (inside a cluster range, at an odd number of K blocks; a range there that
    names a cluster tile is ``None``), of :data:`K1_PARITY_RANGES_BY_SM` (which comes before the next table) or of
    :data:`K1_ORDERED_RANGES_BY_SM`, else ``None``."""
    kw = dict(arch=arch, L=L, M=d_in, N=d_out, K=T, out_f32=f32, sm_count=sms)
    if cake_backend.k1_dsm_plan(**kw) is not None:
        odd = cake_backend.k1_dsm_odd_block_plan(**kw)
        return odd if odd and len(odd) == 4 else None
    parity = cake_backend.k1_parity_plan(**kw)
    if parity:
        return parity
    return cake_backend.k1_ordered_plan(**kw)


def _k1_wgrad_route(arch, sms, d_in, d_out, T, f32):
    """``(kind, block_n, splits)`` of the planner's weight-gradient route of a row: the stream-K table first
    (``"sk"``), then the cluster tables (``"dsm"``; at an odd number of K blocks in a range of
    :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` that table's program, ``"dsm"`` for a cluster tile, ``"odd"`` or
    ``"det"``), the ranges of
    :data:`K1_ORDERED_RANGES_BY_SM` (``"ordered"``) and the split tables (``"pair"`` / ``"det"``)."""
    sk = cake_backend.K1_WGRAD_SK_RULES_BY_SM.get((arch, sms), {})
    template = sk.get((d_in, d_out, T, "f32" if f32 else "bf16"))
    if template is not None:
        return ("sk", int(template.rsplit("_n", 1)[1].split("_")[0]), None)
    kw = dict(arch=arch, L=1, M=d_in, N=d_out, K=T, out_f32=f32, sm_count=sms)
    dsm = cake_backend.k1_dsm_plan(**kw)
    if dsm is not None:
        odd = cake_backend.k1_dsm_odd_block_plan(**kw)
        if odd is not None and len(odd) == 2:
            return ("dsm", *odd)  # the odd-block table's cluster tile
        if odd is not None:
            return ("odd", *odd[:2]) if odd else ("det", None, None)
        return ("dsm", *dsm)
    ordered = cake_backend.k1_ordered_plan(**kw)
    if ordered is not None:
        return ("ordered", ordered[0], ordered[1])
    plan = cake_backend.k1_plan(**kw)
    return ("det", None, None) if plan is None else ("pair", *plan)


def test_k1_wgrad_changed_rows_take_their_route():
    """The weight-gradient rows whose route changed (:data:`K1_WGRAD_CHANGED_ROWS`) take it on their part: the
    stream-K program, two CTAs of the cluster program, the CTA-pair program of a range of
    :data:`K1_ORDERED_RANGES_BY_SM` or :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` or at two splits (or one), or the
    deterministic program."""
    for (arch, sms), rows in K1_WGRAD_CHANGED_ROWS.items():
        for (d_in, d_out, dt, T), want in rows.items():
            got = _k1_wgrad_route(arch, sms, d_in, d_out, T, dt == "f32")
            assert got == want, (arch, sms, d_in, d_out, dt, T, got)


#: The parts the no-varying-plan check covers (the bench parts and the other SM counts of the tables).
K1_ROUTE_PARTS = (
    ("sm_100a", 148),
    ("sm_100a", 152),
    ("sm_103a", 148),
    ("sm_103a", 152),
    ("sm_107a", 208),
    ("sm_107a", 212),
    ("sm_107a", 216),
)


def test_k1_no_order_varying_plan_on_bench_rows():
    """On every part of :data:`K1_ROUTE_PARTS`, no weight-gradient row of the bench's ``small-output`` class (BF16
    and FP32 out, its token counts) plans the CTA-pair program at three or more splits in the arrival order (partials
    added in the order they finish): the planner gives the stream-K program, the cluster program, a fixed split
    order, one or two splits, or the deterministic program.  The tables (:data:`K1_ORDERED_RANGES_BY_SM`,
    :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM` at its odd block counts) hold no such range.  The router
    weight gradient of the bench's ``router`` class plans no stream-K partition with more than two partials per
    element: its row runs the SIMT program, the deterministic program (at the split count of its ``"det_s<n>"``
    window or the deterministic package's), the K-order program of a ``"ksum_S<n>"`` window (a tile's partials added
    in K order), or a partition of at most two partials."""
    bench = _bench()
    rows = [
        spec
        for spec in bench.row_table().values()
        if bench.row_class(spec) in ("small-output", "router") and spec["op"] == "wgrad"
    ]
    assert rows
    checked = 0
    for spec in rows:
        for arch, sms in K1_ROUTE_PARTS:
            if spec["family"] == "router":
                H, T = spec["K"], spec["T"]
                part = cake_backend.router_partition(
                    "nn_t", arch=arch, M=H, N=256, K=T, sm_count=sms
                )
                rule = cake_backend._router_rule_spec(
                    "nn_t", arch=arch, sm_count=sms, M=H, N=256, K=T
                )
                if (
                    part is not None
                    and cake_backend.router_ksum_spec_splits(rule) is not None
                ):
                    # the K-order program of the part
                    assert cake_backend.router_ksum_template(arch, sms), (arch, sms)
                elif part is not None:
                    m_tiles, n_tiles, _ = cake_backend._router_grid(H, 256, T)
                    assert cake_backend.router_at_most_two_partials(
                        m_tiles // 2, n_tiles, part[0]
                    ), (arch, sms, H, T, part)
            else:
                route = _k1_wgrad_route(
                    arch,
                    sms,
                    spec["K"],
                    spec["N"],
                    spec["T"],
                    spec["out_dtype"] == "f32",
                )
                assert route[0] != "pair" or route[2] <= 2, (arch, sms, spec, route)
                if route[0] in ("ordered", "odd") and route[2] > 2:
                    order = _k1_fixed_program(
                        arch,
                        sms,
                        spec["K"],
                        spec["N"],
                        spec["T"],
                        spec["out_dtype"] == "f32",
                    )[3]
                    assert order != "arrival", (arch, sms, spec, route)
            checked += 1
    assert checked == len(rows) * len(K1_ROUTE_PARTS)
    for rows_ in cake_backend.K1_ORDERED_RANGES_BY_SM.values():
        for ranges in rows_.values():
            for _, _, (_, splits, _, order) in ranges:
                assert order != "arrival" or splits <= 2
    for rows_ in cake_backend.K1_DSM_ODD_BLOCK_RANGES_BY_SM.values():
        for ranges in rows_.values():
            for _, _, program in ranges:
                if program is not None and len(program) == 4:
                    assert program[3] != "arrival" or program[1] <= 2
    # nor at any token count of a table on its part: the tables carry no CTA-pair verdict of three or more splits
    for choice in cake_backend.K1_SPLIT_RULES.values():
        assert choice is None or choice[1] <= 2
    for table in (
        cake_backend.K1_SPLIT_RULES_BY_SM,
        cake_backend.K1_SPLIT_OVERRIDES_BY_SM,
    ):
        for rows_ in table.values():
            assert all(c is None or c[1] <= 2 for c in rows_.values())
    for table in (
        cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM,
        cake_backend.K1_TWO_SPLIT_RANGES_BY_SM,
    ):
        for rows_ in table.values():
            assert all(v[1] <= 2 for ranges in rows_.values() for _, _, v in ranges)


class _EveryTemplate(dict):
    """A registry that holds every template (the walk plans what the tables name, whatever this checkout builds)."""

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return "cake_dense_projection_gemm_" + "0" * 20

    def get(self, key, default=None):
        return self[key]


#: The K1 weight-gradient tables keyed by ``(arch, sm_count)`` whose rows are ``(arch, L, d_in, d_out, dtype)``.
_K1_WGRAD_ROW_TABLES = (
    "K1_SPLIT_RULES_BY_SM",
    "K1_SPLIT_RULES_BY_SM_TOKEN_RANGES",
    "K1_SPLIT_RANGE_OVERRIDES_BY_SM",
    "K1_TWO_SPLIT_RANGES_BY_SM",
    "K1_SPLIT_OVERRIDES_BY_SM",
    "K1_CHAIN_RANGES_BY_SM",
    "K1_ONE_SPLIT_RANGES_BY_SM",
    "K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM",
    "K1_DSM_RANGES_BY_SM",
    "K1_DSM_ODD_BLOCK_RANGES_BY_SM",
    "K1_ORDERED_RANGES_BY_SM",
)


def _walk_parts():
    """Every ``(arch, sm_count)`` key of a ``*_BY_SM`` table of the backend and every :data:`ROUTER_SM_COUNT`, one SM
    more and one fewer, and two SM counts of no table per architecture."""
    parts = set(cake_backend.ROUTER_SM_COUNT.items())
    for name in dir(cake_backend):
        table = getattr(cake_backend, name)
        if name.isupper() and "_BY_SM" in name and isinstance(table, dict):
            parts |= {
                k
                for k in table
                if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str)
            }
    parts |= {(a, s + d) for a, s in set(parts) for d in (-1, 1)}
    parts |= {(a, s) for a in cake_backend.ROUTER_SM_COUNT for s in (132, 160)}
    return sorted(parts)


def _range_edges(value) -> set:
    """The token counts next to the ranges of a table value (``((lo, hi, ...), ...)``; ``hi`` may be ``None``)."""
    edges = set()
    if isinstance(value, tuple):
        for r in value:
            if isinstance(r, tuple) and len(r) >= 2 and isinstance(r[0], int):
                for t in (r[0], r[1]):
                    if isinstance(t, int):
                        edges |= {t - 1, t, t + 1}
    return {t for t in edges if 1 <= t <= 32769}


def test_no_default_plan_varies_run_to_run(monkeypatch):
    """Every default plan of the shipped tables repeats bit for bit: on every part of a table (every ``(arch,
    sm_count)`` entry of a ``*_BY_SM`` table and :data:`ROUTER_SM_COUNT`, one SM more and fewer, and SM counts of no
    table), every K1 weight-gradient row of a table (:data:`K1_SPLIT_RULES`, the ``*_BY_SM`` split, chain, cluster,
    odd-block and ordered tables, the stream-K and deterministic-program tables) in BF16 and FP32 out, and every router
    row (:data:`ROUTER_RULES`, :data:`ROUTER_RULES_BY_SM`: the forward, the input gradient and the weight gradient), at
    the edges of every range and window of the row and at the bench token counts, plans no launch that is not
    ``reproducible`` (the K1 CTA-pair program at three or more splits in the arrival order, a K2 partition of more than
    two partials per element), with every template registered.  The forward and input-gradient program tables name
    programs that add each element's whole K in one chain (the deterministic plan's bits), the stream-K table one that
    sums a split tile in unit order: no plan of theirs varies."""
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        monkeypatch.setitem(cake_jit.KERNELS, arch, _EveryTemplate())
    parts = _walk_parts()
    common = {1024, 4096, 8192, 16172, 16231, 16384, 32768}
    common |= set(cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS)
    # K1 weight-gradient rows and their token counts
    rows: dict = {}
    for key in cake_backend.K1_SPLIT_RULES:
        rows.setdefault(key, set())
    for name in _K1_WGRAD_ROW_TABLES:
        for part_rows in getattr(cake_backend, name).values():
            for key, value in part_rows.items():
                rows.setdefault(key, set()).update(_range_edges(value))
    for name in ("K1_WGRAD_SK_RULES_BY_SM", "K1_WGRAD_DET_RULES_BY_SM"):
        for (arch, _), part_rows in getattr(cake_backend, name).items():
            for d_in, d_out, T, dt in part_rows:
                rows.setdefault((arch, 1, d_in, d_out, dt), set()).add(T)
    seen = {"dsm": 0, "fixed": 0, "pair": 0, "plans": 0}
    for (arch, L, d_in, d_out, dt), tokens in sorted(rows.items()):
        Ts = sorted(tokens | common)
        Tmax = max(Ts)
        f32 = dt == "f32"
        X = torch.empty(L, Tmax, d_in, dtype=torch.bfloat16)
        G = torch.empty(L, Tmax, d_out, dtype=torch.bfloat16)
        out = torch.empty(
            L, d_out, d_in, dtype=torch.float32 if f32 else torch.bfloat16
        )
        if L == 1:
            X, G, out = X[0], G[0], out[0]
        for part_arch, sms in parts:
            if part_arch != arch:
                continue
            for T in Ts:
                A = X[..., :T, :].transpose(-1, -2)
                plan, *_ = cake_backend.plan_atomic_gemm(
                    A, G[..., :T, :], out, transposed_out=True, arch=arch, sm_count=sms
                )
                if plan is None:
                    continue
                where = (arch, sms, L, d_in, d_out, dt, T, plan.template)
                assert plan.reproducible, where
                seen["plans"] += 1
                seen["dsm"] += plan.dsm
                seen["fixed"] += plan.split_order != "arrival"
                seen["pair"] += not plan.dsm and plan.splits <= 2
    assert all(seen.values()), seen
    # router rows (fp32 views, 256 experts or the row's own)
    router_rows: dict = {}
    for table in (cake_backend.ROUTER_RULES, *cake_backend.ROUTER_RULES_BY_SM.values()):
        for key, rule in table.items():
            router_rows.setdefault(key, set()).update(_router_spec_tokens(rule))
    kinds = {"simt": 0, "ksum": 0, "partition": 0}
    for (arch, layout, hidden, experts), tokens in sorted(router_rows.items()):
        Ts = sorted(t for t in tokens | common if t <= 32769)
        Tmax = max(Ts)
        X = torch.empty(Tmax, hidden)
        G = torch.empty(Tmax, experts)
        W = torch.empty(experts, hidden)
        for part_arch, sms in parts:
            if part_arch != arch:
                continue
            for T in Ts:
                if layout == "nn_t":
                    args = (G[:T].t(), X[:T], torch.empty(experts, hidden))
                elif layout == "kk":
                    args = (X[:T], W.t(), torch.empty(T, experts))
                else:
                    args = (G[:T], W, torch.empty(T, hidden))
                plan, _ = cake_backend.plan_atomic_router(
                    *args, arch=arch, sm_count=sms
                )
                if plan is None:
                    continue
                where = (arch, sms, layout, hidden, experts, T, plan.template)
                assert plan.reproducible, where
                if plan.simt is not None:
                    kinds["simt"] += 1
                elif plan.ksum:
                    kinds["ksum"] += 1
                else:
                    kinds["partition"] += 1
    assert kinds["simt"] and kinds["ksum"], kinds


def test_k1_forced_arrival_row_alerts_through_the_convenience_wgrad(
    monkeypatch, torch_determinism
):
    """A table row forced to four splits in the arrival order (set here: no shipped table holds one,
    ``test_no_default_plan_varies_run_to_run``) alerts through ``cake_dense_gemm_nondet_train_projection_wgrad()`` under
    ``torch.use_deterministic_algorithms(True)`` (CPU binding, nothing loaded or launched): the planning call and the
    call from the cached key raise before the program; with ``warn_only=True`` the call warns and launches."""
    entry_fn = _cpu_k1_launches(monkeypatch, "sm_103a")
    # the 64-wide FP32 transposed-store program is registered on sm_103a (its record's argument plan binds the call)
    assert cake_backend.atomic_k1_template(64, True) in cake_jit.KERNELS["sm_103a"]
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    monkeypatch.setitem(
        cake_backend.K1_ORDERED_RANGES_BY_SM,
        ("sm_103a", 152),
        {("sm_103a", 1, 4096, 32, "f32"): ((1000, 1000, (64, 4, 1, "arrival")),)},
    )
    bf16, T = torch.bfloat16, 1000
    G, X = torch.empty(T, 32, dtype=bf16), torch.empty(T, 4096, dtype=bf16)
    out = torch.zeros(32, 4096)

    def call():
        return cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)

    torch.use_deterministic_algorithms(True)
    for _ in range(2):  # the planning call, then the call from the cached key
        with pytest.raises(RuntimeError, match="at 4 splits"):
            call()
    ((_, entry),) = cake_backend._K1_ATOMIC_CACHE.items()
    plan = entry.plan
    assert (plan.block_n, plan.splits, plan.split_order) == (64, 4, "arrival")
    assert entry.order_varying and not plan.reproducible
    assert not entry_fn.calls
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="at 4 splits"):
        assert call() is out
    assert len(entry_fn.calls) == 1


#: One row of each K1 program table per part: ``(table, part, key)`` (``key`` as in the table; ``None``: the first row
#: of the table that no table before it in the routing holds).
_K1_PROGRAM_TABLE_ROWS = (
    ("K1_FORWARD_PROGRAM_RULES_BY_SM", ("sm_103a", 152), (4096, 512, 16172)),
    ("K1_FORWARD_RULES_BY_SM", ("sm_103a", 152), (4096, 512, 2048)),
    ("K1_DGRAD_PROGRAM_RULES_BY_SM", ("sm_100a", 148), (6144, 576, 16172, "f32")),
    ("K1_DGRAD_NVJET_RULES_BY_SM", ("sm_103a", 152), (4096, 32, 16231, "bf16")),
    ("K1_DGRAD_RULES_BY_SM", ("sm_100a", 152), None),
    ("K1_WGRAD_SK_RULES_BY_SM", ("sm_103a", 152), (5120, 1024, 16172, "bf16")),
)


def _k1_program_row_call(table, key):
    """``(generic views, transposed_out, entry point name, the entry point's views, out)`` of a row of a K1 program
    table on CPU tensors (never written: nothing is launched)."""
    bf16, f32 = torch.bfloat16, torch.float32
    if table.startswith("K1_FORWARD"):
        K, N, T = key
        X, W = torch.empty(T, K, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out = torch.empty(T, N, dtype=bf16)
        return (X, W.t()), False, "forward", (X, W), out
    if table.startswith("K1_DGRAD"):
        K, N, T, dt = key
        G, W = torch.empty(T, N, dtype=bf16), torch.empty(N, K, dtype=bf16)
        out = torch.empty(T, K, dtype=f32 if dt == "f32" else bf16)
        return (G, W), False, "dgrad", (G, W), out
    d_in, d_out, T, dt = key
    G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
    out = torch.empty(d_out, d_in, dtype=f32 if dt == "f32" else bf16)
    return (X.t(), G), True, "wgrad", (G, X), out


def test_k1_generic_entries_take_the_program_tables(monkeypatch):
    """The generic prepare (``cake_dense_gemm_nondet_train_projection_prepare`` without knobs), the generic convenience
    call and the weight gradient's prepare route a row of each K1 program table of its part to the convenience entry
    point of its operation (so the same program and bits): the forward ``(X, W.T)`` of
    :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` / :data:`K1_FORWARD_RULES_BY_SM`, the input gradient ``(G, W)`` of
    :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` / :data:`K1_DGRAD_NVJET_RULES_BY_SM` / :data:`K1_DGRAD_RULES_BY_SM` and the
    weight gradient ``(X.T, G)`` with the transposed store of :data:`K1_WGRAD_SK_RULES_BY_SM`.  The prepared launch is
    a :class:`CakeDenseGemmNondetProgramLaunch` that names the table and the row's template and whose ``launch()``
    makes the entry point's call with the same views and PDL flag; the same views at another token count are not
    routed."""
    entries = {}
    for op in ("forward", "dgrad", "wgrad"):
        entries[op] = Recorder()
        monkeypatch.setattr(
            cake_backend, f"cake_dense_gemm_nondet_train_projection_{op}", entries[op]
        )
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    order = [t for t, _, _ in _K1_PROGRAM_TABLE_ROWS]
    for i, (table, (arch, sms), key) in enumerate(_K1_PROGRAM_TABLE_ROWS):
        rows = getattr(cake_backend, table)[(arch, sms)]
        if key is None:
            before = [getattr(cake_backend, t).get((arch, sms), {}) for t in order[:i]]
            key = sorted(k for k in rows if not any(k in b for b in before))[0]
        value = rows[key]
        facts = det_backend.DeviceFacts(
            index=0, arch=arch, sm_count=sms, l2_bytes=126 << 20
        )
        monkeypatch.setattr(det_backend, "device_facts", lambda device=None, f=facts: f)
        generic, transposed, op, views, out = _k1_program_row_call(table, key)
        entry = entries[op]
        routed = cake_backend._k1_table_call(*generic, out, transposed)
        assert routed is not None and routed[:2] == (table, key), (table, routed)
        kw = dict(transposed_out=True) if transposed else {}
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            *generic, out, **kw
        )
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProgramLaunch)
        own = table in cake_backend._K1_OWN_PROGRAM_TABLES
        if own:
            template = value
            assert template in cake_jit.KERNELS[arch], (table, template)
        else:
            template = None
            if isinstance(value, str):
                template = det_backend._TEMPLATE_PREFIX + value
        assert (prepared.table, prepared.template, prepared.own) == (
            table,
            template,
            own,
        )
        launches = [
            prepared.launch,
            lambda generic=generic, out=out, kw=kw: (
                cake_backend.cake_dense_gemm_nondet_train_projection(
                    *generic, out, **kw
                )
            ),
        ]
        if transposed:
            wgrad = cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                *views, out
            )
            assert (wgrad.table, wgrad.template) == (table, template)
            launches.append(wgrad.launch)
        for launch in launches:
            entry.calls.clear()
            launch()
            ((args, kwargs),) = entry.calls
            assert [_geometry(a) for a in args[:2]] == [_geometry(v) for v in views]
            assert args[2] is out and kwargs == {"enable_pdl": None}, (table, kwargs)
        # the PDL flag given when prepared reaches the entry point
        cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            *generic, out, enable_pdl=False, **kw
        ).launch()
        assert entry.calls[-1][1] == {"enable_pdl": False}, table
        # the same views at 8 tokens fewer: no row of a table
        if transposed:
            other = (generic[0][:, :-8], generic[1][:-8]), out
        else:
            other = (generic[0][:-8], generic[1]), out[:-8]
        assert cake_backend._k1_table_call(*other[0], other[1], transposed) is None


#: The ``"det_s<n>"`` windows of the router tables: ``(arch, sm_count) -> {hidden: (min_tokens, max_tokens, splits)}``.
ROUTER_DET_WINDOWS = {
    ("sm_100a", 152): {6144: (16129, 16384, 6)},
    ("sm_103a", 152): {6144: (16129, 16384, 6)},
    ("sm_100a", 148): {4096: (16384, 16384, 16), 6144: (16129, 16384, 6)},
    ("sm_103a", 148): {6144: (16129, 16384, 6)},
    ("sm_107a", 212): {5120: (16172, 16384, 12)},
}


def test_router_det_split_windows():
    """:func:`router_det_splits` gives the split count of the ``"det_s<n>"`` window that holds the token count of a
    weight-gradient row (256 experts) on the parts of :data:`ROUTER_DET_WINDOWS` and ``None`` one token outside it, at
    every other hidden size, on every other SM count and for the other layout classes; :func:`router_partition` and
    :func:`router_simt` plan nothing there (the deterministic program).  No router table holds a head/tail window."""
    for table in (cake_backend.ROUTER_RULES, *cake_backend.ROUTER_RULES_BY_SM.values()):
        for rule in table.values():
            specs = [s for _, _, s in rule] if isinstance(rule, tuple) else [rule]
            assert not any(cake_backend.router_head_tail(s) for s in specs), rule
    assert cake_backend.router_det_spec_splits("det_s9") == 9
    for s in ("det_s", "det_9", "H0.5x3", "waves", 6, None, "simt_n64m64s2"):
        assert cake_backend.router_det_spec_splits(s) is None, s
    parts = [(a, s) for a in ("sm_100a", "sm_103a") for s in (148, 150, 152, 160)]
    parts += [("sm_107a", s) for s in (208, 212, 216)]
    seen = 0
    for arch, sms in parts:
        own = ROUTER_DET_WINDOWS.get((arch, sms), {})
        for H in (4096, 5120, 6144):
            window = own.get(H)
            Ts = {1000, 16128, 16129, 16171, 16172, 16231, 16383, 16384, 16385, 20480}
            for T in sorted(Ts):
                kw = dict(arch=arch, M=H, N=256, K=T, sm_count=sms)
                want = None
                if window is not None and window[0] <= T <= window[1]:
                    want = window[2]
                assert cake_backend.router_det_splits("nn_t", **kw) == want, (
                    arch,
                    sms,
                    H,
                    T,
                )
                assert cake_backend.router_det_splits("kk", **kw) is None
                if want is not None:
                    assert cake_backend.router_partition("nn_t", **kw) is None
                    assert cake_backend.router_simt("nn_t", **kw) is None
                    seen += 1
    assert seen == sum(
        sum(1 for T in {16129, 16171, 16172, 16231, 16383, 16384} if lo <= T <= hi)
        for windows in ROUTER_DET_WINDOWS.values()
        for lo, hi, _ in windows.values()
    )


@pytest.mark.parametrize(
    "arch, sms, H, T",
    [
        ("sm_103a", 152, 6144, 16231),
        ("sm_100a", 152, 6144, 16384),
        ("sm_100a", 148, 4096, 16384),
        ("sm_107a", 212, 5120, 16172),
    ],
)
def test_router_det_split_rows_plan_the_det_program(monkeypatch, arch, sms, H, T):
    """A weight-gradient call of a ``"det_s<n>"`` row runs the deterministic program at ``n`` splits: the router entry
    point's cached launch (:data:`_K2_SPLITK_CACHE`) and the convenience call bind the deterministic plan of ``n``
    splits, ``cake_dense_gemm_nondet_train_router_plan`` returns it, and an explicit split count keeps its own; one
    token outside the window the deterministic package's split count of the row."""
    _cpu_k2_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=sms, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    n = ROUTER_DET_WINDOWS[(arch, sms)][H][2]
    G, X = torch.empty(T, 256), torch.empty(T, H)
    out = torch.empty(256, H)
    assert cake_backend._k2_det_splits(G.t(), X, arch, sms) == n
    assert (
        cake_backend.plan_atomic_router(G.t(), X, out, arch=arch, sm_count=sms)[0]
        is None
    )
    assert cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, out) is out
    ((_, entry),) = cake_backend._K2_SPLITK_CACHE.items()
    assert entry.plan.splits == n and entry.k2 is not None
    want = det_backend.cake_dense_gemm_det_train_router_plan(G.t(), X, out, splits=n)[0]
    assert entry.plan == want
    got = cake_backend.cake_dense_gemm_nondet_train_router_plan(G.t(), X, out)[0]
    assert got == want
    cake_backend._K2_SPLITK_CACHE.clear()
    cake_backend._K2_ENTRY_CACHE.clear()
    # the convenience path of the generic entry point
    assert cake_backend.cake_dense_gemm_nondet_train_router(G.t(), X, out) is out
    plans = [e.plan for e in cake_backend._K2_SPLITK_CACHE.values()]
    assert plans == [want]
    # one token outside the window: the deterministic package's split count
    lo, hi, _ = ROUTER_DET_WINDOWS[(arch, sms)][H]
    t = hi + 1
    Gt, Xt = torch.empty(t, 256), torch.empty(t, H)
    assert cake_backend._k2_det_splits(Gt.t(), Xt, arch, sms) is None
    plain = det_backend.cake_dense_gemm_det_train_router_plan(Gt.t(), Xt, out)[0]
    assert (
        cake_backend.cake_dense_gemm_nondet_train_router_plan(Gt.t(), Xt, out)[0]
        == plain
    )


def _part_or_skip():
    """``(device, arch, sm_count)`` of a device the deterministic package registers programs for (else skip)."""
    device = _det_device_or_skip()
    facts = det_backend.device_facts(device)
    return device, facts.arch, facts.sm_count


def _repeats_into_nan(run, out, what, reruns=RERUNS):
    """``reruns`` more calls of ``run(dst)`` into NaN-poisoned outputs give ``out``'s bits."""
    for _ in range(reruns):
        again = torch.full_like(out, float("nan"))
        run(again)
        torch.cuda.synchronize()
        assert torch.equal(again, out), what


def _meets_the_budget_into_nan(run, out, ref, what, reruns=RERUNS):
    """``reruns`` more calls of ``run(dst)`` into NaN-poisoned outputs each meet the K1 budget against ``ref`` (an
    order-varying program: its bits may differ from ``out``'s)."""
    for _ in range(reruns):
        again = torch.full_like(out, float("nan"))
        run(again)
        torch.cuda.synchronize()
        _k1_check(again, ref, what)


def test_gpu_k1_dgrad_rows_run_their_program_with_the_det_bits():
    """On a part of a table of :data:`K1_DGRAD_RULES_BY_SM`: the input gradient of every row (not a row of
    :data:`K1_DGRAD_PROGRAM_RULES_BY_SM` or :data:`K1_DGRAD_NVJET_RULES_BY_SM`, which run this package's programs)
    binds the table's program, gives the
    deterministic package's input gradient bit for bit (the deterministic plan's bits), meets the deterministic
    package's budget against FP64 and repeats bit for bit over ``RERUNS`` calls into NaN-poisoned outputs.  A part
    without such rows skips."""
    device, arch, sms = _part_or_skip()
    own = {
        **cake_backend.K1_DGRAD_PROGRAM_RULES_BY_SM.get((arch, sms), {}),
        **cake_backend.K1_DGRAD_NVJET_RULES_BY_SM.get((arch, sms), {}),
    }
    rows = {
        key: value
        for key, value in cake_backend.K1_DGRAD_RULES_BY_SM.get((arch, sms), {}).items()
        if key not in own
    }
    if not rows:
        pytest.skip(f"no K1 input-gradient row of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 170)
    ran = 0
    for (K, N, T, dt), program in sorted(rows.items()):
        what = f"dgrad {K} {N} {T} {dt} {program}"
        out_dtype = torch.float32 if dt == "f32" else torch.bfloat16
        G, W = _bf16((T, N), device, g), _bf16((N, K), device, g)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full((T, K), float("nan"), device=device, dtype=out_dtype)

        def run(dst, G=G, W=W):
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, dst)

        run(out)
        torch.cuda.synchronize()
        ((_, entry),) = cake_backend._K1_DET_CACHE.items()
        assert entry.k1 is not None, what
        assert entry.plan.template == det_backend._TEMPLATE_PREFIX + program, what
        det_out = torch.full_like(out, float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_dgrad(G, W, det_out)
        torch.cuda.synchronize()
        assert torch.equal(out, det_out), what
        _k1_check(out, G.double() @ W.double(), what)
        _repeats_into_nan(run, out, what)
        ran += 1
    print(
        f"K1_DGRAD_ROWS {arch} sm_count={sms}: {ran} rows ran their program with the deterministic bits"
    )
    assert ran == len(rows)


def test_gpu_k1_dgrad_promo_ranges_keep_the_det_bits():
    """On a part of a table of :data:`K1_DGRAD_PROMO_RANGES_BY_SM`: the input gradient of every row at the edges and
    inside of each range binds the deterministic plan of the row with the range's TMA L2 promotion (the deterministic
    plan's program), gives the deterministic package's input gradient bit for bit, meets the deterministic package's
    budget against FP64 and repeats bit for bit over ``RERUNS`` calls into NaN-poisoned outputs.  A part without such
    rows skips."""
    device, arch, sms = _part_or_skip()
    rows = cake_backend.K1_DGRAD_PROMO_RANGES_BY_SM.get((arch, sms), {})
    if not rows:
        pytest.skip(f"no K1 input-gradient promotion range of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 176)
    ran = 0
    for (K, N, dt), ranges in sorted(rows.items()):
        out_dtype = torch.float32 if dt == "f32" else torch.bfloat16
        W = _bf16((N, K), device, g)
        for lo, hi, promo in ranges:
            for T in sorted({lo, (lo + hi) // 2, hi}):
                what = f"dgrad {K} {N} {T} {dt} promo {promo}"
                G = _bf16((T, N), device, g)
                cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
                out = torch.full((T, K), float("nan"), device=device, dtype=out_dtype)

                def run(dst, G=G, W=W):
                    cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(
                        G, W, dst
                    )

                run(out)
                torch.cuda.synchronize()
                ((_, entry),) = cake_backend._K1_DET_CACHE.items()
                assert entry.k1 is not None and entry.plan.promo == promo, what
                det_out = torch.full_like(out, float("nan"))
                det_backend.cake_dense_gemm_det_train_projection_dgrad(G, W, det_out)
                torch.cuda.synchronize()
                assert torch.equal(out, det_out), what
                _k1_check(out, G.double() @ W.double(), what)
                _repeats_into_nan(run, out, what)
                ran += 1
    print(
        f"K1_DGRAD_PROMO {arch} sm_count={sms}: {ran} rows ran the promoted plan with the deterministic bits"
    )
    assert ran


def test_gpu_k1_dgrad_nvjet_rows_keep_the_det_bits():
    """On a part of a table of :data:`K1_DGRAD_NVJET_RULES_BY_SM`: the input gradient of every row binds this package's
    nvjet program on the deterministic package's plan of :data:`K1_DGRAD_NVJET_DET_PROGRAM`, gives the deterministic
    package's input gradient bit for bit, meets the deterministic package's budget against FP64 and repeats bit for bit
    over ``RERUNS`` calls into NaN-poisoned outputs; the first row runs the program by kernel name.  A part without
    such rows skips."""
    device, arch, sms = _part_or_skip()
    rows = cake_backend.K1_DGRAD_NVJET_RULES_BY_SM.get((arch, sms), {})
    if not rows:
        pytest.skip(f"no K1 input-gradient nvjet row of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 174)
    det_program = cake_backend.K1_DGRAD_NVJET_DET_PROGRAM
    det_template = det_backend._TEMPLATE_PREFIX + det_program
    for i, ((K, N, T, dt), template) in enumerate(sorted(rows.items())):
        what = f"dgrad {K} {N} {T} {dt} {template}"
        module = cake_jit.KERNELS[arch][template]
        G, W = _bf16((T, N), device, g), _bf16((N, K), device, g)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full((T, K), float("nan"), device=device, dtype=torch.bfloat16)

        def run(dst, G=G, W=W):
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, dst)

        run(out)
        torch.cuda.synchronize()
        ((_, entry),) = cake_backend._K1_DET_CACHE.items()
        assert entry.k1 is not None and entry.module_name == module, what
        assert entry.plan.template == det_template, what
        det_out = torch.full_like(out, float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_dgrad(G, W, det_out)
        torch.cuda.synchronize()
        assert torch.equal(out, det_out), what
        _k1_check(out, G.double() @ W.double(), what)
        _repeats_into_nan(run, out, what)
        if i == 0:
            _assert_ran(lambda G=G, W=W, out=out: run(out, G, W), module)


def test_gpu_k1_forward_named_rows_keep_the_det_bits():
    """On a part whose table of :data:`K1_FORWARD_RULES_BY_SM` names a program: the forward of every such row (not a row
    of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM`, which runs this package's program) binds it, gives the deterministic package's forward bit for bit, meets the BF16 budget against FP64 and repeats bit for
    bit over ``RERUNS`` calls into NaN-poisoned outputs.  A part without such rows skips."""
    device, arch, sms = _part_or_skip()
    own = cake_backend.K1_FORWARD_PROGRAM_RULES_BY_SM.get((arch, sms), {})
    rows = {
        key: value
        for key, value in cake_backend.K1_FORWARD_RULES_BY_SM.get(
            (arch, sms), {}
        ).items()
        if isinstance(value, str) and key not in own
    }
    if not rows:
        pytest.skip(
            f"no named K1 forward row of {arch} on {sms} SMs outside the program table"
        )
    g = torch.Generator(device=device).manual_seed(SEED + 171)
    for (K, N, T), program in sorted(rows.items()):
        what = f"forward {K} {N} {T} {program}"
        X, W = _bf16((T, K), device, g), _bf16((N, K), device, g)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full((T, N), float("nan"), device=device, dtype=torch.bfloat16)

        def run(dst, X=X, W=W):
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, dst)

        run(out)
        torch.cuda.synchronize()
        ((_, entry),) = cake_backend._K1_DET_CACHE.items()
        assert entry.k1 is not None, what
        assert entry.plan.template == det_backend._TEMPLATE_PREFIX + program, what
        det_out = torch.full_like(out, float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_forward(X, W, det_out)
        torch.cuda.synchronize()
        assert torch.equal(out, det_out), what
        _k1_check(out, X.double() @ W.double().t(), what)
        _repeats_into_nan(run, out, what)


def test_gpu_k1_program_rows_keep_the_det_bits():
    """On a part of a table of :data:`K1_FORWARD_PROGRAM_RULES_BY_SM` / :data:`K1_DGRAD_PROGRAM_RULES_BY_SM`: the
    forward / input gradient of every row (in the input-gradient token window, the token counts of
    :data:`_KD_WINDOW_SAMPLE`) runs this package's program (no cached deterministic plan; the program bound in
    :data:`_K1_PROGRAM_CACHE`, which the repeats launch), gives the deterministic package's result bit for bit, meets
    the deterministic package's budget against FP64 and repeats bit for bit over ``RERUNS`` calls into NaN-poisoned
    outputs.  A part without rows skips."""
    device, arch, sms = _part_or_skip()
    fwd = cake_backend.K1_FORWARD_PROGRAM_RULES_BY_SM.get((arch, sms), {})
    dgrad_rows = cake_backend.K1_DGRAD_PROGRAM_RULES_BY_SM.get((arch, sms), {})
    dgrad = {
        key: template
        for key, template in dgrad_rows.items()
        if key[2] not in _KD_WINDOW or key[2] in _KD_WINDOW_SAMPLE
    }
    if not fwd and not dgrad:
        pytest.skip(f"no K1 program row of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 173)
    for (K, N, T), template in sorted(fwd.items()):
        what = f"forward {K} {N} {T} {template}"
        X, W = _bf16((T, K), device, g), _bf16((N, K), device, g)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full((T, N), float("nan"), device=device, dtype=torch.bfloat16)

        def run(dst, X=X, W=W):
            cake_backend.cake_dense_gemm_nondet_train_projection_forward(X, W, dst)

        run(out)
        torch.cuda.synchronize()
        assert not cake_backend._K1_DET_CACHE, what
        ((_, entry),) = cake_backend._K1_PROGRAM_CACHE.items()
        assert entry.module_name == cake_jit.KERNELS[arch][template], what
        assert (
            cake_backend._k1_forward_program_of(X, W, out)
            == cake_jit.KERNELS[arch][template]
        ), what
        ref = torch.full_like(out, float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_forward(X, W, ref)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), what
        _k1_check(out, X.double() @ W.double().t(), what)
        _repeats_into_nan(run, out, what)
    for (K, N, T, dt), template in sorted(dgrad.items()):
        what = f"dgrad {K} {N} {T} {dt} {template}"
        G, W = _bf16((T, N), device, g), _bf16((N, K), device, g)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        dtype = torch.float32 if dt == "f32" else torch.bfloat16
        out = torch.full((T, K), float("nan"), device=device, dtype=dtype)

        def run(dst, G=G, W=W):
            cake_backend.cake_dense_gemm_nondet_train_projection_dgrad(G, W, dst)

        run(out)
        torch.cuda.synchronize()
        assert not cake_backend._K1_DET_CACHE, what
        ((_, entry),) = cake_backend._K1_PROGRAM_CACHE.items()
        assert entry.module_name == cake_jit.KERNELS[arch][template], what
        assert (
            cake_backend._k1_dgrad_program_of(G, W, out)
            == cake_jit.KERNELS[arch][template]
        ), what
        ref = torch.full_like(out, float("nan"))
        det_backend.cake_dense_gemm_det_train_projection_dgrad(G, W, ref)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), what
        _k1_check(out, G.double() @ W.double(), what)
        _repeats_into_nan(run, out, what)


def test_gpu_k1_wgrad_changed_rows_repeat_and_meet_the_budget():
    """On a part of :data:`K1_WGRAD_CHANGED_ROWS`: the weight gradient of every changed row takes its route (the
    stream-K program of :data:`K1_WGRAD_SK_RULES_BY_SM` with no cached plan, or its :data:`_K1_ATOMIC_CACHE` entry:
    the cluster program, the CTA-pair program at one or two splits or in the split order of its range of
    :data:`K1_ORDERED_RANGES_BY_SM` or :data:`K1_DSM_ODD_BLOCK_RANGES_BY_SM`, or the deterministic program -- then the deterministic package's weight gradient
    bit for bit; a row of :data:`K1_WGRAD_DET_RULES_BY_SM` the bits of the deterministic package's prepared launch of
    the table's program), meets the deterministic package's budget against FP64 and repeats bit for bit over
    ``RERUNS`` calls into NaN-poisoned outputs -- a four-split arrival-order range meets the budget on each of those
    calls instead (its bits may vary).  A part without changed rows skips."""
    device, arch, sms = _part_or_skip()
    rows = K1_WGRAD_CHANGED_ROWS.get((arch, sms), {})
    if not rows:
        pytest.skip(f"no changed K1 weight-gradient row of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 172)
    for (d_in, d_out, dt, T), (kind, block_n, splits) in sorted(rows.items()):
        what = f"wgrad {d_in} {d_out} {dt} {T} {kind}"
        f32 = dt == "f32"
        G, X = _bf16((T, d_out), device, g), _bf16((T, d_in), device, g)
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        out = torch.full(
            (d_out, d_in),
            float("nan"),
            device=device,
            dtype=torch.float32 if f32 else torch.bfloat16,
        )

        def run(dst, G=G, X=X):
            cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, dst)

        run(out)
        torch.cuda.synchronize()
        if kind == "sk":
            # the stream-K program runs straight from its table: no cached plan of the split tables
            assert not cake_backend._K1_ATOMIC_CACHE, what
            assert cake_backend._k1_wgrad_sk_program_of(G, X, out, "auto", None), what
            _k1_check(out, G.double().t() @ X.double(), what)
            _repeats_into_nan(run, out, what)
            continue
        ((_, entry),) = cake_backend._K1_ATOMIC_CACHE.items()
        if kind == "det":
            assert entry.plan is None and entry.k1 is not None, what
            program = cake_backend.K1_WGRAD_DET_RULES_BY_SM.get((arch, sms), {}).get(
                (d_in, d_out, T, dt)
            )
            ref = torch.full_like(out, float("nan"))
            if program is None:
                det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X, ref)
            else:
                assert (
                    entry.det_plan.template == det_backend._TEMPLATE_PREFIX + program
                ), what
                A, B, transposed = det_backend.wgrad_views(G, X, arch=arch, out_f32=f32)
                knobs = dict(cake_backend.k1_program_knobs(program))
                # the symbol forms the prepared entry does not take keep their plain values in the table's programs
                plain = ("sk_first", "lw", "mg", "sw")
                assert not any(knobs.pop(name) for name in plain), what
                launch = det_backend.cake_dense_gemm_det_train_projection_prepare(
                    A, B, ref, transposed_out=transposed, **knobs
                )
                assert launch.template == entry.det_plan.template, what
                launch.launch()
            torch.cuda.synchronize()
            assert torch.equal(out, ref), what
        varying = False
        if kind != "det":
            plan = entry.plan
            assert plan is not None, what
            ordered = _k1_fixed_program(arch, sms, d_in, d_out, T, f32)
            if ordered is not None:
                assert (plan.chains, plan.split_order) == tuple(ordered[2:]), what
            # only a range of K1_ORDERED_RANGES_BY_SM or K1_DSM_ODD_BLOCK_RANGES_BY_SM adds three or more splits in
            # the arrival order
            varying = ordered is not None and not plan.reproducible
            assert plan.reproducible or varying, what
            assert (plan.dsm, plan.block_n, plan.splits) == (
                kind == "dsm",
                block_n,
                splits,
            ), what
        ref64 = G.double().t() @ X.double()
        _k1_check(out, ref64, what)
        if varying:
            _meets_the_budget_into_nan(run, out, ref64, what)
        else:
            _repeats_into_nan(run, out, what)


def test_gpu_k1_parity_ranges_run_at_their_edges():
    """On a part with ranges in :data:`K1_PARITY_RANGES_BY_SM` for its SM count (or above
    :data:`K1_PARITY_RANGES_ABOVE_SM`, those of that count): at both ends of every range of a row and at an even and an
    odd number of K blocks inside it, the weight gradient binds the program of the token count's parity
    (:func:`k1_parity_plan`) -- one split, its chains added in order -- meets the deterministic package's budget against
    FP64 and repeats bit for bit over ``RERUNS`` calls into NaN-poisoned outputs; where the range gives the
    deterministic program it binds the deterministic package's plan and gives that package's bits.  A part without
    ranges skips."""
    device, arch, sms = _part_or_skip()
    part = cake_backend.K1_PARITY_RANGES_BY_SM.get((arch, sms))
    top = cake_backend.K1_PARITY_RANGES_ABOVE_SM.get(arch)
    if part is None and top is not None and sms > top:
        part = cake_backend.K1_PARITY_RANGES_BY_SM.get((arch, top))
    if not part:
        pytest.skip(f"no K1 parity range of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 173)
    ran = 0
    for (_, L, d_in, d_out, dt), ranges in sorted(part.items()):
        f32 = dt == "f32"
        cases = set()
        for lo, hi, _, _ in ranges:
            mid = (lo + hi) // 2
            cases |= {T for T in (lo, hi, mid, mid + 64) if lo <= T <= hi}
        for T in sorted(cases):
            want = cake_backend.k1_parity_plan(
                arch=arch, L=L, M=d_in, N=d_out, K=T, out_f32=f32, sm_count=sms
            )
            if want is None:
                continue
            what = f"parity {d_in} {d_out} {dt} {T} {want}"
            G, X = _bf16((T, d_out), device, g), _bf16((T, d_in), device, g)
            cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
            out = torch.full(
                (d_out, d_in),
                float("nan"),
                device=device,
                dtype=torch.float32 if f32 else torch.bfloat16,
            )

            def run(dst, G=G, X=X):
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, dst)

            run(out)
            torch.cuda.synchronize()
            ((_, entry),) = cake_backend._K1_ATOMIC_CACHE.items()
            if want == ():
                assert entry.plan is None and entry.det_plan is not None, what
                ref = torch.full_like(out, float("nan"))
                det_backend.cake_dense_gemm_det_train_projection_wgrad(G, X, ref)
                torch.cuda.synchronize()
                assert torch.equal(out, ref), what
                continue
            plan = entry.plan
            assert plan is not None and entry.k1 is not None, what
            assert (
                plan.block_n,
                plan.splits,
                plan.chains,
                plan.split_order,
            ) == want, what
            assert plan.reproducible, what
            _k1_check(out, G.double().t() @ X.double(), what)
            _repeats_into_nan(run, out, what)
            ran += 1
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    print(
        f"K1_PARITY_RANGES {arch} sm_count={sms}: {ran} parity programs ran at range edges"
    )
    assert ran


_RESIDENT_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(128) cake_test_resident(long long cycles) {
  extern __shared__ unsigned char hold[];  // one block per SM: the dynamic shared memory fills the SM
  long long start = clock64();
  if (threadIdx.x == 0) hold[0] = 0;
  while (clock64() - start < cycles) {
    __nanosleep(100);
  }
}
"""


def _resident_kernel(device, sms, us):
    """``(launch, stream, keep)``: ``launch()`` enqueues ``sms`` blocks of 128 threads, each with most of an SM's
    shared memory (one block per SM), that spin for at least ``us`` microseconds, on ``stream`` (NVRTC to
    ``compute_80`` PTX through ``cuda.bindings``; ``keep`` holds the module and the argument buffers)."""
    import numpy as np
    from cuda.bindings import driver, nvrtc

    def ok(result):
        status, *values = result if isinstance(result, tuple) else (result,)
        assert int(getattr(status, "value", status)) == 0, status
        return values[0] if len(values) == 1 else values

    prog = ok(
        nvrtc.nvrtcCreateProgram(_RESIDENT_SOURCE.encode(), b"resident.cu", 0, [], [])
    )
    opts = [b"--gpu-architecture=compute_80"]
    ok(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    ptx = b" " * ok(nvrtc.nvrtcGetPTXSize(prog))
    ok(nvrtc.nvrtcGetPTX(prog, ptx))
    torch.zeros(1, device=device)  # the primary context, current on this thread
    module = ok(driver.cuModuleLoadData(ptx))
    function = ok(driver.cuModuleGetFunction(module, b"cake_test_resident"))
    props = torch.cuda.get_device_properties(device)
    smem = int(getattr(props, "shared_memory_per_block_optin", 232448)) - 1024
    attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    ok(driver.cuFuncSetAttribute(function, attr, smem))
    stream = torch.cuda.Stream(device)
    arg = np.array([int(us * 1e-6 * 2.2e9)], dtype=np.int64)  # cycles at up to 2.2 GHz
    params = np.array([arg.ctypes.data], dtype=np.uint64)

    def launch():
        ok(
            driver.cuLaunchKernel(
                function, sms, 1, 1, 128, 1, 1, smem,
                driver.CUstream(stream.cuda_stream), params.ctypes.data, 0,
            )
        )  # fmt: skip

    return launch, stream, (module, arg, params)


def test_gpu_k1_wgrad_stream_k_completes_next_to_a_resident_kernel():
    """On a part of :data:`K1_WGRAD_SK_RULES_BY_SM`: the stream-K weight gradient of the part's first row, launched
    right after and right before a kernel that holds 32 SMs for 5 ms on another stream, in eager launches and in CUDA
    graph replays, completes within a host deadline with the bits of the launch alone and every flag word zero: the
    unit that sums a split tile waits only for lower units, so a launch whose CTA pairs are not all resident at once
    still completes.  A part without rows skips."""
    device, arch, sms = _part_or_skip()
    rows = cake_backend.K1_WGRAD_SK_RULES_BY_SM.get((arch, sms), {})
    if not rows:
        pytest.skip(f"no stream-K weight-gradient row of {arch} on {sms} SMs")
    d_in, d_out, T, dt = min(rows)
    g = torch.Generator(device=device).manual_seed(SEED + 174)
    G, X = _bf16((T, d_out), device, g), _bf16((T, d_in), device, g)
    cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
    dtype = torch.float32 if dt == "f32" else torch.bfloat16
    out = torch.full((d_out, d_in), float("nan"), device=device, dtype=dtype)

    def run():
        cake_backend.cake_dense_gemm_nondet_train_projection_wgrad(G, X, out)

    run()
    torch.cuda.synchronize()
    assert cake_backend._k1_wgrad_sk_program_of(G, X, out, "auto", None)
    ref = out.clone()
    launch, blocker, keep = _resident_kernel(device, 32, 5000)
    stream = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        run()
    stream.wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    def flags():
        return sum(
            int(arena.buffers["k1_tickets"].view(torch.int32).count_nonzero())
            for arena in cake_backend._ATOMIC_ARENAS.values()
            if "k1_tickets" in arena.buffers
        )

    for mode in ("eager", "graph"):
        for first in ("resident", "stream-K"):
            for _ in range(3):
                what = f"{arch} {sms} {d_in} {d_out} {T} {dt} {mode}, {first} first"
                out.fill_(float("nan"))
                blocker.wait_stream(stream)
                if first == "resident":
                    launch()
                (graph.replay if mode == "graph" else run)()
                if first == "stream-K":
                    launch()
                done = torch.cuda.Event()
                done.record(stream)
                deadline = time.monotonic() + 30.0
                while not done.query():
                    if time.monotonic() > deadline:
                        pytest.fail(f"{what}: no completion within 30 s")
                    time.sleep(1e-3)
                stream.wait_stream(blocker)
                torch.cuda.synchronize()
                assert torch.equal(out, ref), what
                assert flags() == 0, what
    del graph, keep


def test_gpu_router_det_split_rows_run_the_det_program():
    """On a part of :data:`ROUTER_DET_WINDOWS`: the router weight gradient (256 experts) of every ``"det_s<n>"``
    window, at its first and last token count and 16231 tokens where the window holds them, gives the deterministic
    package's weight gradient at ``n`` splits bit for bit, meets the router budget against FP64 and repeats bit for bit
    over ``RERUNS`` calls into NaN-poisoned outputs.  A part without such a window skips."""
    device, arch, sms = _part_or_skip()
    windows = ROUTER_DET_WINDOWS.get((arch, sms), {})
    if not windows:
        pytest.skip(f"no router det-split window of {arch} on {sms} SMs")
    g = torch.Generator(device=device).manual_seed(SEED + 173)
    for H, (lo, hi, n) in sorted(windows.items()):
        for T in sorted({lo, hi} | ({16231} if lo <= 16231 <= hi else set())):
            what = f"router wgrad H{H} T{T} det_s{n}"
            G = torch.randn(T, 256, device=device, generator=g)
            X = torch.randn(T, H, device=device, generator=g)
            cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
            out = torch.full((256, H), float("nan"), device=device)

            def run(dst, G=G, X=X):
                cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, dst)

            run(out)
            torch.cuda.synchronize()
            ref = torch.full_like(out, float("nan"))
            det_backend.cake_dense_gemm_det_train_router_wgrad(G, X, ref, splits=n)
            torch.cuda.synchronize()
            assert torch.equal(out, ref), what
            exact = G.double().t() @ X.double()
            assert bool(torch.isfinite(out).all()), what
            assert _rel_fro(out, exact) <= ROUTER_REL_FRO_MAX, (
                what,
                _rel_fro(out, exact),
            )
            _repeats_into_nan(run, out, what)


def test_gpu_router_ksum_rows_repeat_and_meet_the_budget(torch_determinism):
    """On a part with a ``"ksum_S<n>"`` window (212-SM sm_107a: hidden 6144 at 16172 to 16231 tokens): the router weight
    gradient at the window's first and last token count runs the K-order program (by kernel name), the convenience
    call and the prepared launch give the same bits, meet the router budget against FP64, repeat bit for bit over
    ``RERUNS`` calls into NaN-poisoned outputs and raise no alert under ``torch.use_deterministic_algorithms(True)``.
    A part without such a window skips."""
    device, arch, sms = _part_or_skip()
    template = cake_backend.router_ksum_template(arch, sms)
    rules = cake_backend.ROUTER_RULES_BY_SM.get((arch, sms), {})
    windows = [
        (H, lo, hi)
        for (rule_arch, layout, H, _), rule in rules.items()
        if rule_arch == arch and layout == "nn_t" and isinstance(rule, tuple)
        for lo, hi, spec in rule
        if cake_backend.router_ksum_spec_splits(spec) is not None
    ]
    if template is None or not windows:
        pytest.skip(f"no router K-order window of {arch} on {sms} SMs")
    module = cake_jit.KERNELS[arch][template]
    g = torch.Generator(device=device).manual_seed(SEED + 176)
    for H, lo, hi in windows:
        for T in sorted({lo, hi}):
            what = f"router wgrad ksum H{H} T{T}"
            G = torch.randn(T, 256, device=device, generator=g)
            X = torch.randn(T, H, device=device, generator=g)
            cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
            out = torch.full((256, H), float("nan"), device=device)

            def run(dst, G=G, X=X):
                cake_backend.cake_dense_gemm_nondet_train_router_wgrad(G, X, dst)

            torch.use_deterministic_algorithms(True)
            with _no_warnings():
                run(out)
            torch.use_deterministic_algorithms(False)
            torch.cuda.synchronize()
            exact = G.double().t() @ X.double()
            assert bool(torch.isfinite(out).all()), what
            assert _rel_fro(out, exact) <= ROUTER_REL_FRO_MAX, (
                what,
                _rel_fro(out, exact),
            )
            prepared = cake_backend.cake_dense_gemm_nondet_train_router_prepare(
                G.t(), X, torch.full_like(out, float("nan"))
            )
            assert prepared.plan.ksum and prepared.plan.reproducible, what
            prepared.launch()
            torch.cuda.synchronize()
            assert torch.equal(prepared.out, out), what
            _repeats_into_nan(run, out, what)
            if lo == T:
                _assert_ran(lambda: run(out), module)


#: The views of :func:`test_gpu_k1_program_rows_take_offset_and_padded_views`: label -> the program table.
_K1_VIEW_TABLES = {
    "fwd": "K1_FORWARD_PROGRAM_RULES_BY_SM",
    "dgrad": "K1_DGRAD_PROGRAM_RULES_BY_SM",
    "nvjet": "K1_DGRAD_NVJET_RULES_BY_SM",
    "wgrad": "K1_WGRAD_SK_RULES_BY_SM",
}


def test_gpu_k1_program_rows_take_offset_and_padded_views():
    """On the device's part, the first row of each K1 program table of the part (:data:`_K1_VIEW_TABLES`): the operands
    column slices of wider buffers at a 16-byte offset (the weight a row slice of a taller one), ``out`` a slice of a
    larger NaN-filled buffer with a padded row stride and a nonzero storage offset.  The entry point of the operation
    still takes the row's table on these views, and its call, the generic prepared launch (a
    :class:`CakeDenseGemmNondetProgramLaunch` of the table; for the weight gradient also its own prepare) and the
    generic convenience call give the bits of the entry point on contiguous copies (this package's own programs by
    kernel name), meet the FP64 budget and leave the rest of the buffer untouched.  A part without program rows
    skips."""
    device, arch, sms = _part_or_skip()
    g = torch.Generator(device=device).manual_seed(SEED + 177)
    bf16, f32 = torch.bfloat16, torch.float32
    entry = {
        "fwd": cake_backend.cake_dense_gemm_nondet_train_projection_forward,
        "dgrad": cake_backend.cake_dense_gemm_nondet_train_projection_dgrad,
        "nvjet": cake_backend.cake_dense_gemm_nondet_train_projection_dgrad,
        "wgrad": cake_backend.cake_dense_gemm_nondet_train_projection_wgrad,
    }
    routed = {
        "fwd": cake_backend._k1_forward_program_of,
        "dgrad": cake_backend._k1_dgrad_program_of,
        "nvjet": cake_backend._k1_dgrad_nvjet_of,
        "wgrad": lambda G, X, o: cake_backend._k1_wgrad_sk_program_of(
            G, X, o, "auto", None
        ),
    }
    ran = 0
    for label, table in _K1_VIEW_TABLES.items():
        rows = getattr(cake_backend, table).get((arch, sms), {})
        if not rows:
            continue
        key, template = sorted(rows.items())[0]
        what = f"{label} {key} {template}"
        cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear()
        if label == "fwd":
            (K, N, T), dtype = key, bf16
            a = _bf16((T, K + 64), device, g)[:, 8 : 8 + K]
            b = _bf16((N + 16, K), device, g, K**-0.5)[16:]
            generic, transposed = (a, b.t()), False
            rows_, cols = T, N
            ref64 = a.double() @ b.double().t()
        elif label in ("dgrad", "nvjet"):
            K, N, T, dt = key
            dtype = f32 if dt == "f32" else bf16
            a = _bf16((T, N + 64), device, g)[:, 8 : 8 + N]
            b = _bf16((N + 16, K), device, g, N**-0.5)[16:]
            generic, transposed = (a, b), False
            rows_, cols = T, K
            ref64 = a.double() @ b.double()
        else:
            d_in, d_out, T, dt = key
            dtype = f32 if dt == "f32" else bf16
            a = _bf16((T, d_out + 64), device, g)[:, 8 : 8 + d_out]  # G
            b = _bf16((T, d_in + 64), device, g)[:, 8 : 8 + d_in]  # X
            generic, transposed = (b.t(), a), True
            rows_, cols = d_out, d_in
            ref64 = a.double().t() @ b.double()
        assert a.storage_offset() and b.storage_offset(), what
        # the bits of the entry point on contiguous copies
        ref = torch.full((rows_, cols), float("nan"), device=device, dtype=dtype)
        entry[label](a.contiguous(), b.contiguous(), ref)
        torch.cuda.synchronize()
        buf, out = _sliced_out(rows_, cols, dtype, device)
        assert out.storage_offset() and out.stride(0) == cols + 24
        assert routed[label](a, b, out) is not None, what
        kw = dict(transposed_out=True) if transposed else {}
        prepared = cake_backend.cake_dense_gemm_nondet_train_projection_prepare(
            *generic, out, **kw
        )
        assert isinstance(prepared, cake_backend.CakeDenseGemmNondetProgramLaunch)
        assert prepared.table == table, (what, prepared.table)
        launches = [
            lambda: entry[label](a, b, out),
            prepared.launch,
            lambda: cake_backend.cake_dense_gemm_nondet_train_projection(
                *generic, out, **kw
            ),
        ]
        if transposed:
            launches.append(
                cake_backend.cake_dense_gemm_nondet_train_projection_wgrad_prepare(
                    a, b, out
                ).launch
            )
        for i, launch in enumerate(launches):
            out.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
            assert torch.equal(out, ref), (what, i)
            _k1_check(out, ref64, f"{what} {i}")
            assert _outside_untouched(buf, cols), (what, i)
        if label != "nvjet":
            _assert_ran(prepared.launch, cake_jit.KERNELS[arch][template])
        ran += 1
    if not ran:
        pytest.skip(f"no K1 program row of {arch} on {sms} SMs")
