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

# The dense projection GEMM package with an explicit numerical mode: the dispatch of every entry point
# (``deterministic=True`` -> the deterministic package's canonical object, ``deterministic=False`` -> the routes of
# ``cake_backend.ROUTES``), PyTorch's deterministic-algorithms rule, the names of the former import path, the K1 and
# K2 planners (the per-architecture planners, the rule tables, the token-count guard, the stream-K partition), the
# registry and file names of this package's programs and its workspace kinds.  The CPU tests launch nothing.  The GPU
# tests (``test_gpu_*``) need a device the deterministic package registers programs for; the ones of this package's
# programs force the split count (they do not depend on what the planner takes on the device), assert by kernel
# name that the program ran, and skip only on a checkout that registers no program of this package for the device.

import ast
import contextlib
import dataclasses
import inspect
import json
import re
import sys
import threading
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

# Backend suite: the packages are named through ``cake_package``; nothing else of flashinfer is imported here.
package = cake_package("dense_projection_gemm")
cake_backend = cake_package("dense_projection_gemm", "cake_backend")
cake_jit = cake_package("dense_projection_gemm", "cake_jit")
det_backend = cake_package("dense_projection_det_gemm", "cake_backend")
det_jit = cake_package("dense_projection_det_gemm", "cake_jit")
# A vendored run imports both packages under other names: bind the deterministic package explicitly (in a
# FlashInfer tree this names the sibling the package resolves by itself).
cake_backend.use_deterministic_backend(det_backend)

# Accuracy gates (the deterministic package's, unchanged by the numerical mode).
BF16_ATOL = BF16_RTOL = 1e-2
F32_REL_FRO_MAX = 5e-5
F32_MAX_ABS_OVER_REF_ABSMAX = 1e-3
ROUTER_REL_FRO_MAX = 2e-6
RERUNS = 20
SEED = 20261031

# The names of ``cake_backend`` before #8046 that are entry points: each takes ``deterministic`` here.
ENTRY_POINTS = (
    "dense_projection_gemm",
    "instance_symbol",
    "plan_dense_projection_gemm",
    "plan_router_fp32_gemm",
    "prepare_dense_projection_gemm",
    "prepare_projection_wgrad",
    "prepare_router_fp32_gemm",
    "projection_dgrad",
    "projection_forward",
    "projection_wgrad",
    "router_dgrad",
    "router_forward",
    "router_fp32_gemm",
    "router_symbol",
    "router_wgrad",
    "split_fp32_to_bf16x3_",
)
# former name -> the deterministic package's canonical name (its alias block, kept by #8046)
FORMER_TO_DET = {
    "GemmPlan": "DetGemmPlan",
    "PreparedGemm": "PreparedDetGemm",
    "PreparedRouterGemm": "PreparedRouterDetGemm",
    "RouterPlan": "RouterDetPlan",
    "dense_projection_gemm": "dense_projection_det_gemm",
    "plan_dense_projection_gemm": "plan_dense_projection_det_gemm",
    "plan_router_fp32_gemm": "plan_router_fp32_det_gemm",
    "prepare_dense_projection_gemm": "prepare_dense_projection_det_gemm",
    "prepare_projection_wgrad": "prepare_projection_det_wgrad",
    "prepare_router_fp32_gemm": "prepare_router_fp32_det_gemm",
    "projection_dgrad": "projection_det_dgrad",
    "projection_forward": "projection_det_forward",
    "projection_wgrad": "projection_det_wgrad",
    "router_dgrad": "router_det_dgrad",
    "router_forward": "router_det_forward",
    "router_fp32_gemm": "router_fp32_det_gemm",
    "router_wgrad": "router_det_wgrad",
    "split_fp32_to_bf16x3_": "split_fp32_to_bf16x3_det_",
}
# Every public name ``cake_backend`` defined before #8046 (FlashInfer c472a430, functions, classes and constants).
# fmt: off
FORMER_MODULE_NAMES = (
    "ARCH_RULE_DEFAULTS", "BLOCK_K", "BLOCK_M", "BLOCK_N_CHOICES", "B_SWZ_CHOICES", "CH_T_COLS",
    "CONVENIENCE_CACHE_SIZE", "CTA_GROUP", "CTA_ROWS_CHOICES", "DeviceFacts", "EPI_MODES", "EPI_WARPS",
    "EXACT_WAVE_BLOCK_N", "GROUP_M", "GemmPlan", "K_TMA_BF16", "K_TWO_SLOTS", "L2_HINTS", "L2_PROMOS",
    "PANEL_BYTES", "PD_CHOICES", "PreparedGemm", "PreparedRouterGemm", "ROUTER_BLOCK_K", "ROUTER_BLOCK_M",
    "ROUTER_BLOCK_N", "ROUTER_SPLITS", "ROW_RULES", "RULE_PARENT", "RouterPlan", "SK_COUNTERS_MIN",
    "SK_DUMMY_BASE", "SK_MIN_ITERS", "SK_SLAB_CHOICES", "SLOT_BYTES", "SMEM_OPT_IN", "SMEM_OVERSIZED",
    "SPLIT_BLOCK_ELEMS", "SPLIT_TEMPLATE", "SUPPORTED_COMPUTE_CAPABILITIES", "W8_FORMS", "WORK_STAGES",
    "a_halves_of", "arch_for", "as_batched", "b_stage_bytes", "bind_launch", "box_rows_of", "check_stream_k_plan",
    "clear_convenience_caches", "convenience_key", "default_block_n", "default_cta_rows", "default_group_m",
    "default_hints", "default_hints_cgrp", "default_pf", "default_promo", "default_stages",
    "dense_projection_gemm", "device_facts", "device_l2_bytes", "device_sm_count", "epi_cols", "epi_mode",
    "epi_slots", "exact_waves", "flat_alias", "fused_split_module", "generated_program_available", "instance_key",
    "instance_symbol", "operand_view", "plan_dense_projection_gemm", "plan_router_fp32_gemm",
    "prepare_dense_projection_gemm", "prepare_projection_wgrad", "prepare_router_fp32_gemm", "projection_dgrad",
    "projection_forward", "projection_wgrad", "require_arch", "router_dgrad", "router_forward",
    "router_fp32_gemm", "router_layout_class", "router_symbol", "router_wgrad", "row_rule", "sk_exact_plan",
    "sk_parts_plan", "sk_sync_plan", "sm_pairs", "sm_units", "smem_limit_for", "split_fp32_to_bf16x3_",
    "staging_bytes", "stream_k_plan", "stt_bytes", "swap_small_m", "wave_working_set", "wave_working_set_cgrp",
    "wgrad_swapped", "wgrad_views",
)
# fmt: on
DET_TOKEN = re.compile(r"(?:^|[^a-z])det(?:[^a-z]|$)", re.IGNORECASE)
# the mode is named by deterministic=
NEGATED_TOKEN = re.compile(r"non_?" + "det", re.IGNORECASE)


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


def _patch_det(monkeypatch, former: str) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(det_backend, FORMER_TO_DET[former], recorder)
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


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_points_take_keyword_only_deterministic_true(name):
    parameter = inspect.signature(getattr(cake_backend, name)).parameters[
        "deterministic"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


def test_the_deterministic_names_map():
    assert {
        k: v for k, v in cake_backend.DETERMINISTIC_NAMES.items() if k in FORMER_TO_DET
    } == FORMER_TO_DET
    for former, canonical in FORMER_TO_DET.items():
        assert getattr(det_backend, former) is getattr(det_backend, canonical)
    for name in ENTRY_POINTS:
        assert getattr(cake_backend, name).__module__ == cake_backend.__name__


@pytest.mark.parametrize("name", FORMER_MODULE_NAMES)
def test_every_former_module_name_resolves(name):
    obj = getattr(cake_backend, name)
    if name in ENTRY_POINTS or name in ("bind_launch", "clear_convenience_caches"):
        assert obj.__module__ == cake_backend.__name__
    else:
        assert obj is getattr(
            det_backend, cake_backend.DETERMINISTIC_NAMES.get(name, name)
        )


def test_former_types_are_the_deterministic_classes():
    assert cake_backend.GemmPlan is det_backend.DetGemmPlan
    assert cake_backend.PreparedGemm is det_backend.PreparedDetGemm
    assert cake_backend.RouterPlan is det_backend.RouterDetPlan
    assert cake_backend.PreparedRouterGemm is det_backend.PreparedRouterDetGemm


@pytest.mark.parametrize(
    "name",
    [
        "projection_det_forward",
        "DetGemmPlan",
        "router_fp32_det_gemm",
        "__wrapped__",
        "no_such_name",
    ],
)
def test_deterministic_spellings_are_not_read_through(name):
    with pytest.raises(AttributeError):
        getattr(cake_backend, name)


def test_the_deterministic_package_resolves_as_a_sibling(monkeypatch):
    """In a FlashInfer tree and in a copy vendored with the deterministic package beside it (the vendored gate's
    layout) the deterministic package resolves as this package's sibling."""
    if not cake_jit.__name__.rpartition(".")[0].rpartition(".")[0]:
        pytest.skip("a top-level copy has no parent package to hold a sibling")
    monkeypatch.setattr(cake_jit, "_det_package", None)
    assert cake_backend.deterministic_backend() is det_backend
    assert cake_jit.deterministic_module("cake_jit") is det_jit


def test_use_deterministic_backend_binds_a_vendored_copy(monkeypatch):
    monkeypatch.setattr(cake_jit, "_det_package", None)
    fake_pkg, fake = (
        ModuleType("vendored_det_pkg"),
        ModuleType("vendored_det_pkg.cake_backend"),
    )
    for name in cake_backend.DETERMINISTIC_NAMES.values():
        setattr(fake, name, Recorder())
    monkeypatch.setitem(sys.modules, "vendored_det_pkg", fake_pkg)
    monkeypatch.setitem(sys.modules, "vendored_det_pkg.cake_backend", fake)
    cake_backend.use_deterministic_backend(fake)
    assert cake_backend.deterministic_backend() is fake
    assert (
        cake_backend.projection_forward("X", "W") is fake.projection_det_forward.result
    )
    assert fake.projection_det_forward.calls == [(("X", "W", None), {})]
    with pytest.raises(TypeError, match="lacks"):
        cake_backend.use_deterministic_backend(ModuleType("not_a_backend.cake_backend"))


# ---------------------------------------------------------------------------
# Dispatch (CPU)
# ---------------------------------------------------------------------------

# entry point -> (positional arguments, keyword arguments, the call the deterministic object receives)
# fmt: off
DEFAULT_CALLS = {
    "dense_projection_gemm": (
        ("A", "B", "out"),
        {"transposed_out": True, "enable_pdl": True},
        (("A", "B", "out"), {"transposed_out": True, "enable_pdl": True}),
    ),
    "plan_dense_projection_gemm": (
        ("A", "B", "out"),
        {"sm_count": 148, "l2_bytes": 1 << 20, "transposed_out": True, "block_n": 128},
        (("A", "B", "out"), {"sm_count": 148, "l2_bytes": 1 << 20, "transposed_out": True, "block_n": 128}),
    ),
    "prepare_dense_projection_gemm": (("A", "B", "out"), {"sk": False}, (("A", "B", "out"), {"sk": False})),
    "projection_forward": (("X", "W"), {}, (("X", "W", None), {})),
    "projection_dgrad": (("G", "W", "out"), {"out_dtype": "f32"}, (("G", "W", "out"), {"out_dtype": "f32"})),
    "projection_wgrad": (
        ("G", "X"),
        {"out_dtype": "f32", "sk": True, "cta_rows": 64},
        (("G", "X", None), {"out_dtype": "f32", "sk": True, "cta_rows": 64}),
    ),
    "prepare_projection_wgrad": (
        ("G", "X", "out"),
        {"cta_rows": 128},
        (("G", "X", "out"), {"sk": "auto", "cta_rows": 128}),
    ),
    "plan_router_fp32_gemm": (("A", "B", "out"), {"splits": 3}, (("A", "B", "out"), {"splits": 3})),
    "prepare_router_fp32_gemm": (("A", "B", "out"), {}, (("A", "B", "out"), {"splits": None})),
    "router_fp32_gemm": (("A", "B", "out"), {"splits": 2}, (("A", "B", "out"), {"splits": 2})),
    "router_forward": (("X", "W"), {}, (("X", "W", None), {})),
    "router_dgrad": (("G", "W", "out"), {}, (("G", "W", "out"), {})),
    "router_wgrad": (("G", "X"), {"splits": 9}, (("G", "X", None), {"splits": 9})),
    "split_fp32_to_bf16x3_": (("src", "parts", "resid"), {}, (("src", "parts", "resid"), {})),
}
# fmt: on


@pytest.mark.parametrize("name", sorted(DEFAULT_CALLS))
def test_the_default_calls_the_deterministic_object(
    monkeypatch, torch_determinism, name
):
    """The default path is the deterministic package's object with the same arguments (so the same bits), with and
    without PyTorch's deterministic mode, and never warns."""
    args, kwargs, expected = DEFAULT_CALLS[name]
    recorder = _patch_det(monkeypatch, name)
    for enabled in (False, True):
        torch.use_deterministic_algorithms(enabled)
        with _no_warnings():
            assert getattr(cake_backend, name)(*args, **kwargs) is recorder.result
            assert (
                getattr(cake_backend, name)(*args, deterministic=True, **kwargs)
                is recorder.result
            )
    assert recorder.calls == [expected] * 4


def test_the_default_router_forward_and_wgrad_keep_the_production_splits(monkeypatch):
    forward, wgrad = (
        _patch_det(monkeypatch, "router_forward"),
        _patch_det(monkeypatch, "router_wgrad"),
    )
    cake_backend.router_forward("X", "W", "out")
    cake_backend.router_wgrad("G", "X", "out", splits=6)
    assert forward.calls == [
        (("X", "W", "out"), {})
    ]  # the deterministic default (4 splits) applies
    assert wgrad.calls == [(("G", "X", "out"), {"splits": 6})]


def test_instance_and_router_symbols():
    key = det_backend.instance_key(a_mn=True, b_mn=True)
    assert cake_backend.instance_symbol(
        key, deterministic=False
    ) == det_backend.instance_symbol(key)
    for a_mn, b_mn, out_t in (
        (False, False, False),
        (False, True, False),
        (True, True, True),
    ):
        assert cake_backend.router_symbol(
            a_mn, b_mn, out_t
        ) == det_backend.router_symbol(a_mn, b_mn, out_t)
    assert (
        cake_backend.router_symbol(False, False, False, deterministic=False)
        == "router_fp32_gemm_atomic_kk"
    )
    assert (
        cake_backend.router_symbol(True, True, True, deterministic=False)
        == "router_fp32_gemm_atomic_nn_t"
    )
    assert cake_backend.router_symbol(
        False, True, False, deterministic=False
    ) == det_backend.router_symbol(False, True, False)


@pytest.mark.parametrize("layout", ["kk", "kn", "nn"])
def test_k1_routes(monkeypatch, layout):
    """``deterministic=False``: the weight-gradient layout reaches this package's split-K path; the forward and
    input-gradient layouts run the deterministic program."""
    det = _patch_det(monkeypatch, "dense_projection_gemm")
    atomic = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k1", atomic)
    A, B, out = _k1_views(layout)
    result = cake_backend.dense_projection_gemm(A, B, out, deterministic=False)
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
        _patch_det(monkeypatch, "projection_forward"),
        _patch_det(monkeypatch, "projection_dgrad"),
    )
    atomic = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k1", atomic)
    X, W, G = torch.zeros(64, 32), torch.zeros(16, 32), torch.zeros(64, 16)
    assert cake_backend.projection_forward(X, W, deterministic=False) is forward.result
    assert cake_backend.projection_dgrad(G, W, deterministic=False) is dgrad.result
    assert not atomic.calls
    out = torch.zeros(16, 32, dtype=torch.bfloat16)
    assert (
        cake_backend.projection_wgrad(G, X, out, deterministic=False) is atomic.result
    )
    assert (
        cake_backend.prepare_projection_wgrad(G, X, out, deterministic=False)
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
        _patch_det(monkeypatch, "projection_wgrad"),
        _patch_det(monkeypatch, "prepare_projection_wgrad"),
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
    """``deterministic=False`` K1 calls that run the deterministic program go through this package's binding
    (``_det_k1``): keyed by the entry point's own tensors, with the deterministic package's convenience call as the
    plan (``det_args``) and as the fallback (``det_call``), the entry point's ``enable_pdl`` (and the generic entry's
    ``weights_static``) passed to both; ``deterministic=True`` calls the deterministic package."""
    names = {"fwd": "projection_forward", "dgrad": "projection_dgrad"}
    det = _patch_det(monkeypatch, names.get(op, "dense_projection_gemm"))
    binding = Recorder()
    monkeypatch.setattr(cake_backend, "_det_k1", binding)
    X, W, G = torch.zeros(64, 32), torch.zeros(16, 32), torch.zeros(64, 16)
    if op == "fwd":
        call = lambda d: cake_backend.projection_forward(  # noqa: E731
            X, W, deterministic=d, enable_pdl=pdl
        )
    elif op == "dgrad":
        call = lambda d: cake_backend.projection_dgrad(  # noqa: E731
            G, W, out_dtype=torch.float32, deterministic=d, enable_pdl=pdl
        )
    else:
        A, B, out = _k1_views(op)
        call = lambda d: cake_backend.dense_projection_gemm(  # noqa: E731
            A, B, out, deterministic=d, enable_pdl=pdl, weights_static=bool(pdl)
        )
    assert call(False) is binding.result and not det.calls
    ((sources, flags, det_args, det_call, enable_pdl, *rest), kwargs) = binding.calls[0]
    assert not kwargs and enable_pdl is pdl
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
    det.calls.clear()
    assert call(True) is det.result and len(det.calls) == 1 and len(binding.calls) == 1


def _same_entry_views(views, a_view, split_src, flags) -> None:
    """``views`` (an entry point's own launch views) are ``(a_view, split_src)`` with the plan flags ``flags``."""
    assert _geometry(views[0]) == _geometry(a_view)
    assert views[0].data_ptr() == a_view.data_ptr() and views[1] is split_src
    assert views[2] == flags


def test_k2_routes(monkeypatch):
    """``deterministic=False``: the router weight gradient and input gradient reach this package's K2 path (the input
    gradient without a split count: its SIMT windows, else the deterministic program), and so does the forward with a
    split count or while some router table gives the forward a verdict; without either the forward runs the
    deterministic program through this package's binding (the plan the planner keeps for a row without a verdict).
    The split stage is the deterministic package's.  Each entry point passes its own tensors as the launch views."""
    dgrad, split = (
        _patch_det(monkeypatch, "router_dgrad"),
        _patch_det(monkeypatch, "split_fp32_to_bf16x3_"),
    )
    gemm = _patch_det(monkeypatch, "router_fp32_gemm")
    atomic, splitk = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic)
    monkeypatch.setattr(cake_backend, "_splitk_k2", splitk)
    T, K, N = 64, 32, 16
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    assert cake_backend.router_forward(X, W, deterministic=False) is splitk.result
    ((args, kwargs),) = splitk.calls
    assert args[0] is X and args[1].equal(W.t()) and args[3] is None and not kwargs
    assert args[5] is None  # enable_pdl: the device's default
    assert tuple(args[2].shape) == (T, N) and args[2].dtype == torch.float32
    _same_entry_views(args[4], X.unsqueeze(0), W, cake_backend._FORWARD_VIEWS)
    assert (
        cake_backend.router_forward(X, W, deterministic=False, splits=2)
        is atomic.result
    )
    monkeypatch.setitem(cake_backend.ROUTER_RULES, ("sm_103a", "kk", K, N), "S2")
    assert cake_backend.router_forward(X, W, deterministic=False) is atomic.result
    assert (
        cake_backend.router_wgrad(G, X, deterministic=False, splits=5) is atomic.result
    )
    assert cake_backend.router_dgrad(G, W, deterministic=False) is atomic.result
    assert not dgrad.calls
    assert (
        cake_backend.split_fp32_to_bf16x3_("s", "p", "r", deterministic=False)
        is split.result
    )
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
        cake_backend.router_fp32_gemm(
            G, W, torch.zeros(T, K), deterministic=False, splits=1
        )
        is atomic.result
    )
    assert (
        atomic.calls[4][1]["splits"] == 1 and atomic.calls[4][1]["enable_pdl"] is None
    )
    assert (
        cake_backend.router_fp32_gemm(X, W.t(), torch.zeros(T, N), deterministic=False)
        is atomic.result
    )
    assert (
        cake_backend.router_fp32_gemm(G.t(), X, torch.zeros(N, K), deterministic=False)
        is atomic.result
    )
    assert all(kw["views"] is None for _, kw in atomic.calls[4:])
    assert not gemm.calls and len(splitk.calls) == 1 and len(atomic.calls) == 7


@pytest.mark.parametrize("pdl", [True, False])
def test_k2_routes_pass_enable_pdl(monkeypatch, pdl):
    """``enable_pdl`` reaches every K2 route of ``deterministic=False``: this package's binding of the deterministic
    program (the forward without a verdict) and this package's K2 path (the input gradient, the generic ``kn`` call,
    the forward with a split count, the weight gradient)."""
    atomic, splitk = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic)
    monkeypatch.setattr(cake_backend, "_splitk_k2", splitk)
    T, K, N = 64, 32, 16
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    kw = dict(deterministic=False, enable_pdl=pdl)
    cake_backend.router_forward(X, W, **kw)
    cake_backend.router_dgrad(G, W, **kw)
    cake_backend.router_fp32_gemm(G, W, torch.zeros(T, K), splits=1, **kw)
    cake_backend.router_forward(X, W, splits=2, **kw)
    cake_backend.router_wgrad(G, X, splits=5, **kw)
    assert [args[5] for args, _ in splitk.calls] == [pdl]
    assert [kwargs["enable_pdl"] for _, kwargs in atomic.calls] == [pdl] * 4


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
    """Take :data:`K1_CHAIN_RANGES_BY_SM` out of the planner: its ranges come before every other K1 table
    (:func:`k1_chain_plan`), so a test of those tables holds them without it (``test_k1_chain_ranges`` holds the
    chain ranges)."""
    monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})


@pytest.mark.parametrize(
    "arch, sm_count, rows",
    [
        ("sm_100a", 148, 28),
        ("sm_103a", 152, 32),
        ("sm_103a", 148, 28),
        ("sm_107a", 216, 26),
        ("sm_107a", 212, 19),
        ("sm_107a", 208, 14),
        ("sm_107a", 210, 0),
    ],
)
def test_k1_planners_take_the_measured_rows(arch, sm_count, rows):
    """The verdicts of the three architectures (on sm_107a with the chain ranges of 216 SMs) take 28 / 32 / 26 of the
    124 weight-gradient rows (the MLA head rows, ``L = 64``, run the deterministic program), at 16172 and 16231 tokens;
    on 148 SMs sm_103a leaves ``4096 -> 48`` to the deterministic program (28 rows).  16172 tokens span an odd number of
    64-token K blocks (253), below the first token count of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` (65538), so the
    FP32 rows of that table run their two-split program there.  On sm_107a the token counts of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` run the deterministic program (seven FP32 rows and one BF16 row at 16231
    tokens on 216 SMs; on 212 and 216 SMs outside their ranges in :data:`K1_CHAIN_RANGES_BY_SM`: 16231 tokens for
    ``4096 -> 512``, ``4096 -> 1024`` and ``5120 -> 512`` on 212 SMs and for ``4096 -> 512`` and ``4096 -> 1024`` on 216
    SMs, where the other five and six (row, token count) pairs run three accumulation chains; on 208 SMs ``4096 -> 512``
    and ``4096 -> 1024`` FP32 at 16231 tokens, where the other six pairs of the four one-split FP32 rows run one split).
    The per-SM-count sm_107a tables and the chain ranges take 19 rows on 212 SMs, the tables 14 on 208 SMs; an SM count
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
    # sm_107a: the 216-SM verdicts from 216 SMs; below, the table of the part's own SM count (208, 212) or none
    assert cake_backend.k1_plan(arch="sm_107a", M=4096, N=32, sm_count=216, **kw) == (
        64,
        4,
    )
    for sms in (208, 212):
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
        assert p107(4096, 32, sms, out_f32=True) == (64, 3)
        # 4096 -> 512 BF16 was measured and missed on every SM count
        assert p107(4096, 512, sms) is None
        # 4096 -> 32, 4096 -> 64, 5120 -> 32 BF16 and 6144 -> 32: no verdict (their deterministic program did not
        # complete)
        for M, N in ((4096, 32), (4096, 64), (5120, 32), (6144, 32)):
            assert p107(M, N, sms) is None
    # 4096 -> 4096 FP32: the deterministic program on 208, 212 and 216 SMs
    assert p107(4096, 4096, 212, out_f32=True) is None
    assert p107(4096, 4096, 208, out_f32=True) is None
    assert p107(4096, 4096, 216, out_f32=True) is None
    # 6144 -> 32 FP32: three splits on 216 SMs, the deterministic program on 208 and 212
    assert p107(6144, 32, 212, out_f32=True) is None
    assert p107(6144, 32, 216, out_f32=True) == (64, 3)
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


def test_k1_small_token_counts_run_the_deterministic_program():
    """``num_splits * 64 <= T + 63``: a token count too short for the split count runs the deterministic program."""
    kw = dict(arch="sm_107a", L=1, M=4096, N=32, out_f32=False, sm_count=216)
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
    token count at which they meet the accuracy budget) and end at one or have no upper bound; the row runs the
    deterministic program at both ends and keeps its verdict (inside the token window, the override of
    :data:`K1_SPLIT_OVERRIDES_BY_SM` for the part's SM count, if any) one token outside a range (inside the token window
    of a per-SM-count table), on a part with fewer SMs than an entry at or above the floor, on the floor's part for an
    entry below it, and above 65536 tokens unless a range has no upper bound.  A token count with an odd number of
    64-token blocks inside an entry of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` for the part runs the deterministic
    program too."""
    table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    odd_table = cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS
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
            return False

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
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key107, (64, 2))
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=216, **kw) == (64, 2)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=210, **kw) is None
    # below that count the part's own table decides (K1_SPLIT_RULES_BY_SM), not the 216-SM verdict
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=212, **kw) == (64, 3)
    monkeypatch.setitem(cake_backend.K1_MIN_SM_COUNT, "sm_107a", 0)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=212, **kw) == (64, 2)


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
    (4096, 32, "f32"): ((1024, 16172), (16384, 65536)),
    (4096, 48, "f32"): ((1024, 16172), (16384, 65536)),
    (4096, 64, "f32"): ((1024, 16172), (16384, 65536)),
    (4096, 512, "f32"): ((512, 512),),
    (4096, 1024, "f32"): ((512, 512),),
    (5120, 32, "f32"): ((1024, 16172), (16384, 65536)),
    (5120, 48, "f32"): ((1024, 16172), (16384, 65536)),
    (5120, 64, "f32"): ((1024, 16172), (16384, 65536)),
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
        # 208 SMs: the deterministic program outside the window; 216 SMs: the one-split verdict up to 40960 tokens,
        # the deterministic program from 40961 tokens up
        rule = tuple(cake_backend.K1_SPLIT_RULES[("sm_107a", 1, M, N, "f32")])
        for T in (960, 1024, 6000, 8192, 32768, 40960):
            assert p(M, N, T, sm_count=216) == rule, (M, N, T)
        for T in (1024, 6000, 8192, 32768):
            assert p(M, N, T, sm_count=208) is None, (M, N, T)
        for T in (40961, 65536):
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


def test_k1_152_sm_two_split_ranges(monkeypatch):
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
    ("sm_100a", 5120, 32, "bf16"): ((49153, None),),
    ("sm_100a", 5120, 48, "bf16"): ((512, 2048),),
    ("sm_100a", 5120, 48, "f32"): ((512, 512),),
    ("sm_100a", 5120, 64, "bf16"): ((512, 2048), (8192, 8192)),
    ("sm_100a", 5120, 64, "f32"): ((512, 1024), (8192, 8192)),
    ("sm_103a", 4096, 32, "bf16"): ((65536, None),),
    ("sm_103a", 4096, 32, "f32"): ((65536, None),),
    ("sm_103a", 5120, 48, "bf16"): ((512, 2048),),
    ("sm_103a", 5120, 48, "f32"): ((512, 2048),),
    ("sm_103a", 5120, 64, "bf16"): ((512, 1024),),
    ("sm_103a", 5120, 64, "f32"): ((512, 1024),),
    ("sm_103a", 5120, 128, "bf16"): ((512, 4096), (32768, 65536)),
    ("sm_103a", 5120, 128, "f32"): ((512, 4096), (32768, 65536)),
    ("sm_103a", 6144, 128, "bf16"): ((512, 8192),),
    ("sm_103a", 6144, 128, "f32"): ((512, 8192),),
}
#: The measured token counts of the 148-SM audit (49153: a point of the FP64 error grid of sm_100a ``5120 -> 32`` BF16,
#: 10 seeds at 32769 to 65025 tokens).
K1_148_MEASURED_TOKENS = (512, 1024, 2048, 4096, 8192, 16231, 32768, 49153, 65536)


def test_k1_exact_sm_count_exclusions(monkeypatch):
    """A row runs the deterministic program at the token counts of its entries in
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` on a part of exactly that SM count (148-SM sm_100a / sm_103a parts),
    and every part of any other SM count (147, 149, 152, 160, 208, 212, 216, 220) plans every row and token count as
    without the table.  Every entry belongs to a row of :data:`K1_SPLIT_RULES` with a program on that part; its ranges
    are closed, disjoint and not adjacent, start at a measured token count and end at one or have no upper bound; one
    token outside a range the row keeps the plan it has without the table."""
    table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM
    got_rows = {
        (arch, k[2], k[3], k[4]): tuple(v)
        for (arch, sms), rows in table.items()
        for k, v in rows.items()
    }
    assert set(table) <= {("sm_100a", 148), ("sm_103a", 148)}
    assert got_rows == K1_148_EXCLUDED_TOKENS

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

    Ts = sorted(
        {T + d for T in K1_148_MEASURED_TOKENS for d in (-1, 0, 1)} | {64, 256, 131072}
    )
    others = (147, 149, 152, 160, 208, 212, 216, 220)
    with_table = plans((148,) + others, Ts)
    monkeypatch.setattr(cake_backend, "K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM", {})
    without = plans((148,) + others, Ts)
    for (key, sms, T), plan in with_table.items():
        if sms != 148:
            assert plan == without[(key, sms, T)], (key, sms, T)
            continue
        spans = table[(key[0], 148)][key]
        inside = any(lo <= T and (hi is None or hi >= T) for lo, hi in spans)
        assert plan == (None if inside else without[(key, sms, T)]), (key, T)
    for (arch, sms), rows in table.items():
        assert sms == 148
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
                assert any(b < T < c for T in K1_148_MEASURED_TOKENS), key
            for lo, hi in spans:
                assert lo in K1_148_MEASURED_TOKENS, key
                assert hi is None or hi in K1_148_MEASURED_TOKENS, key
                assert without[(key, 148, lo)] is not None, key
                assert with_table[(key, 148, lo)] is None, key


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
    ("sm_107a", 1, 4096, 32, "bf16"): (512,),
    ("sm_107a", 1, 4096, 48, "f32"): (512,),
    ("sm_107a", 1, 4096, 64, "f32"): (512,),
    ("sm_107a", 1, 5120, 32, "bf16"): (512,),
    ("sm_107a", 1, 5120, 32, "f32"): (512,),
    ("sm_107a", 1, 5120, 48, "f32"): (512,),
    ("sm_107a", 1, 5120, 64, "f32"): (512,),
    ("sm_107a", 1, 6144, 32, "bf16"): (512,),
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
    tokens (:data:`K1_SPLIT_OVERRIDES_BY_SM`), except at 16231 tokens (its 216-SM entry of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`: the deterministic program); one token outside that window and on 217 SMs it
    keeps its :data:`K1_SPLIT_RULES` verdict (256-wide tiles at one split) up to 40960 tokens, on 215 SMs (no table) it
    runs the deterministic program, and on 212 SMs its own tables decide (the 192-wide row-major program at one split in
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

    for T in (16172, 16200, 16300, 16384):
        assert plan(216, T) == (192, 1), T
        assert plan(217, T) == (256, 1), T
        assert plan(212, T) == (192, 1), T
        assert plan(215, T) is None, T
    # 16231 tokens: the deterministic program from 216 SMs up and on 208 / 212 SMs
    for sms in (208, 212, 215, 216, 217, 280):
        assert plan(sms, 16231) is None, sms
    two_212 = K1_212_TWO_SPLIT_RANGES[(d_in, d_out, "f32")]
    for T in (1024, 8192, 16171, 16385, 32768, 40960):
        assert plan(216, T) == (256, 1), T
        assert plan(217, T) == (256, 1), T
        want = [v for lo, hi, v in two_212 if lo <= T <= hi]
        assert plan(212, T) == (want[0] if want else None), T
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
    T = 16384
    G = torch.zeros(T, d_out, dtype=torch.bfloat16)
    X = torch.zeros(T, d_in, dtype=torch.bfloat16)
    out = torch.zeros(d_out, d_in, dtype=torch.float32)
    _register_k1(monkeypatch, arch)
    plan_, *_, reason = cake_backend.plan_atomic_gemm(
        X.t(), G, out, transposed_out=True, arch=arch, sm_count=216
    )
    assert isinstance(plan_, cake_backend.AtomicGemmPlan) and reason == ""
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
    # verdict there; a table of another SM count does not act on 216 SMs (4096 -> 32 FP32: three splits on 216 SMs)
    key32 = (arch, 1, 4096, 32, "f32")
    assert cake_backend.K1_SPLIT_RULES[key32] == (64, 3)
    monkeypatch.setattr(
        cake_backend,
        "K1_SPLIT_OVERRIDES_BY_SM",
        {(arch, 216): {key32: None}, (arch, 220): {key32: (128, 2)}},
    )
    assert plan(216, 16384, N=32) is None and plan(216, 16385, N=32) == (64, 3)
    assert plan(220, 16384, N=32) == (128, 2) and plan(221, 16384, N=32) == (64, 3)


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
    # 4096 -> 1024: at 16231 tokens only; 212 SMs (their table) and 216 SMs (the 216-SM override) keep the 192-wide
    # row-major program at 16384, 208 SMs the 256-wide tiles
    for sms in (208, 212, 216):
        assert p(4096, 1024, sms, 16231) is None, sms
    assert p(4096, 1024, 208, 16384) == (256, 1)
    for sms in (212, 216):
        assert p(4096, 1024, sms, 16384) == (192, 1), sms
    # 4096 -> 48 and 4096 -> 64: entries on 208 and 216 SMs, none on 212
    for N in (48, 64):
        assert p(4096, N, 212, 16231) == (64, 3) and p(4096, N, 212, 16384) == (64, 3)
        for sms in (208, 216):
            assert p(4096, N, sms, 16231) is None and p(4096, N, sms, 16384) is None
    # 5120 -> 48 and 5120 -> 64: 16231 tokens on 208, 212 and 216 SMs; 16384 on 216 SMs, and for 5120 -> 48 on 208
    for N in (48, 64):
        for sms in (208, 212, 216):
            assert p(5120, N, sms, 16231) is None, (N, sms)
        assert p(5120, N, 216, 16384) is None, N
        assert p(5120, N, 212, 16384) == (64, 3), N
    assert p(5120, 48, 208, 16384) is None
    assert p(5120, 64, 208, 16384) == (64, 3)
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
    # 5120 -> 32 keeps its verdict at both token counts; 4096 -> 32 too, except at 16231 tokens on 216 SMs
    for sms in (208, 212, 216):
        for T in (16231, 16384):
            assert p(5120, 32, sms, T) == (64, 3), (sms, T)
            if (sms, T) != (216, 16231):
                assert p(4096, 32, sms, T) == (64, 3), (sms, T)
    assert p(4096, 32, 216, 16231) is None
    # 4096 -> 4096: the deterministic program on 208, 212 and 216 SMs
    for sms in (208, 212, 216):
        assert p(4096, 4096, sms, 16172) is None


#: The first token count at which the sm_107a FP32 rows ``4096 / 5120 -> 512 / 1024`` run the deterministic program on
#: parts of 216 SMs or more: their one-split program met the FP32 accuracy budget (relative Frobenius error at most
#: 5e-5 and largest absolute error at most 1e-3 of the largest reference magnitude, against FP64) on every seed at
#: 40960 tokens and missed it from 45056 tokens up.
K1_SM107A_FP32_BUDGET_TOKENS = 40961
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
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`) and keep their one-split program below (961 to 40960 tokens included).  The
    208-SM and 212-SM tables hold only at 16172 to 16384 tokens and at 512 tokens on 212 SMs
    (:data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`), so those parts run the deterministic program at these token counts,
    except the 212-SM two-split ranges at either number of 64-token K blocks (``K1_212_TWO_SPLIT_RANGES``); the other
    entries of these rows lie in that window.  No other sm_107a row has a range without an upper bound, and sm_100a /
    sm_103a have no verdict for these rows.  These are the tables under :data:`K1_CHAIN_RANGES_BY_SM` (taken out here),
    whose ranges run three accumulation chains on 212 and 216 SMs."""
    T0 = K1_SM107A_FP32_BUDGET_TOKENS
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
        assert cake_backend.K1_SPLIT_RULES[key] == choice
        assert [e for e in excluded[key] if e[2] is None] == [(216, T0, None)]
        for _, lo, hi in excluded[key]:
            assert hi is None or 16172 <= lo <= hi <= 16384, key
        for sms in (216, 217, 280):
            for T in (512, 704, 961, 1024, 8192, 32768, T0 - 1):
                assert p(M, N, sms, T) == choice, (M, N, sms, T)
            for T in (T0, 45056, 65536, 65537, 131072):
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
    ("sm_103a", 1, 4096, 48, "f32"): ((152, 65538, None),),
}


def test_k1_odd_block_exclusions(monkeypatch):
    """A row of :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` runs the deterministic program at a token count of an entry's
    range that spans an odd number of 64-token K blocks, on a part of the entry's SM count or more where the entry is at
    or above the row's SM-count floor and on exactly that SM count below it; at an even number and below the range the
    row keeps the route it takes without the table.  Every entry names an FP32 two-split row with a verdict on its part,
    starts at 65538 tokens (above the largest measured token count at which the two-split program met the FP32 accuracy
    budget, 65537) and has no upper bound; BF16 rows have no entry.  Below 65538 tokens these rows run their two-split
    program at either number of K blocks."""
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
        (("sm_100a", 1, 5120, 48, "f32"), 152, 98305),
        (("sm_103a", 1, 4096, 32, "f32"), 149, 65665),
    ):
        assert got[(key, part, T)] is None, (key, part, T)
        assert p(key, part, T) == (64, 2), (key, part, T)
    # below 65538 tokens odd block counts take the two-split program with the table in place (1025, 4097, 16173 and
    # 65537 tokens); sm_107a ``8192 -> 128`` FP32 has no entry
    monkeypatch.undo()
    table = cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS
    assert table == K1_ODD_BLOCK_EXCLUDED_TOKENS
    for key, part, T, want in (
        (("sm_100a", 1, 4096, 32, "f32"), 148, 1025, (64, 2)),
        (("sm_100a", 1, 4096, 64, "f32"), 152, 65537, (64, 2)),
        (("sm_100a", 1, 5120, 32, "f32"), 148, 16173, (64, 2)),
        (("sm_103a", 1, 4096, 32, "f32"), 148, 4097, (64, 2)),
        (("sm_103a", 1, 4096, 48, "f32"), 152, 16173, (64, 2)),
        (("sm_107a", 1, 8192, 128, "f32"), 212, 4097, (128, 2)),
        (("sm_107a", 1, 8192, 128, "f32"), 216, 1025, (128, 2)),
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
    # (216 SMs: 4096 -> 1024 at 16384 tokens the 216-SM override, 5120 -> 1024 at 32768 tokens one chain)
    assert failing
    monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})
    for (key, sms, T), plan in failing.items():
        _, L, M, N, _ = key
        want = cake_backend.k1_plan(
            arch="sm_107a", L=L, M=M, N=N, K=T, out_f32=True, sm_count=sms
        )
        assert plan == want, (key, sms, T)
    p216 = dict(arch="sm_107a", L=1, out_f32=True, sm_count=216)
    assert failing[(("sm_107a", 1, 4096, 1024, "f32"), 216, 16384)] == (192, 1)
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
    (``AtomicGemmPlan.chains``, template ``..._c3``, no workspace and no tickets): the 128-wide and 256-wide
    transposed-store programs and the 192-wide row-major-store program, inside a 212-SM two-split range too; past the
    chain range the two-split range plans two splits of one chain.  An explicit ``splits`` / ``block_n`` runs one chain;
    ``chains=3`` names the three-chain program of any row at one split; a chain count outside :data:`K1_CHAINS_CHOICES`,
    ``chains=3`` with more than one split or on fewer than three K blocks raise ``ValueError``; a BF16 three-chain
    program is not registered."""
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
        assert isinstance(plan, cake_backend.AtomicGemmPlan) and reason == ""
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
    """The three-chain programs ship on the architectures of their one-chain programs: the 128-wide and 256-wide FP32
    transposed-store programs on sm_100a, sm_103a and sm_107a, the 192-wide FP32 row-major-store program on sm_107a;
    each is the one-chain program's build with ``chains=3``."""
    want = {
        "dense_proj_gemm_atomic_nn_n128_f32_t_c3": ("sm_100a", "sm_103a", "sm_107a"),
        "dense_proj_gemm_atomic_nn_n256_f32_t_c3": ("sm_100a", "sm_103a", "sm_107a"),
        "dense_proj_gemm_atomic_nn_n192_f32_c3": ("sm_107a",),
    }
    got = {}
    for arch, kernels in cake_jit.KERNELS.items():
        for template in kernels:
            if template.endswith(("_c2", "_c3")):
                got.setdefault(template, []).append(arch)
    assert {t: tuple(sorted(a)) for t, a in got.items()} == want
    for template, arches in want.items():
        one = template[: -len("_c3")]
        for arch in arches:
            chained = cake_jit.MODULES[cake_jit.KERNELS[arch][template]]
            single = cake_jit.MODULES[cake_jit.KERNELS[arch][one]]
            assert chained["build"]["module"] == single["build"]["module"]
            assert chained["build"]["kwargs"] == {
                **single["build"]["kwargs"],
                "chains": 3,
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
        assert isinstance(plan, cake_backend.AtomicGemmPlan) and reason == ""
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
        assert isinstance(plan, cake_backend.AtomicGemmPlan) and reason == ""
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
    only where the deterministic plan (6 splits per tile) fills its last wave below 0.8; the deterministic split
    count is the deterministic package's."""
    assert det_backend.ROUTER_SPLITS["nn_t"] == cake_backend.ROUTER_DET_WGRAD_SPLITS
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
    # weight gradient at T = 16231 (508 K blocks): "H0.5x4" on sm_107a at hidden 5120 (108 head + 432 tail units on
    # 216 SMs; 106 + 424 on 212 and 104 + 416 on 208, ROUTER_RULES_BY_SM) and at hidden 6144 on 212-SM parts; the
    # deterministic plan at every hidden size on sm_100a / sm_103a (their head/tail windows missed the accuracy rule,
    # FAILED_ROUTER_WINDOWS), at hidden 6144 on the other sm_107a parts and at sm_107a hidden 4096
    for arch in ("sm_100a", "sm_103a"):
        for hidden in (4096, 5120, 6144):
            assert cake_backend.ROUTER_RULES[(arch, "nn_t", hidden, 256)] == (
                ROUTER_WGRAD_WINDOWS.get((arch, hidden)) or None
            )
            assert (
                cake_backend.router_partition(
                    "nn_t", arch=arch, sm_count=152, M=hidden, N=256, K=16231
                )
                is None
            )
    # the partitions of those windows, forced: "H1x5p44" (44 head units, then 5 tail units per CTA pair) at hidden
    # 4096 on sm_100a and hidden 6144 on sm_103a, "H1x4p44" (four tail units per pair) at hidden 4096 on sm_103a from
    # 20480 tokens (640 K blocks), "H0.5x3" (76 head + 228 tail units) at hidden 5120
    with monkeypatch.context() as m:
        for (arch, layout, hidden), windows in FAILED_ROUTER_WINDOWS.items():
            m.setitem(cake_backend.ROUTER_RULES, (arch, layout, hidden, 256), windows)
        assert cake_backend.router_partition(
            "nn_t", arch="sm_103a", sm_count=152, M=6144, N=256, K=16231
        ) == (44 + 5 * 76, 22, round(24 * 508 * 44 / 76))
        assert cake_backend.router_partition(
            "nn_t", arch="sm_100a", sm_count=152, M=4096, N=256, K=16231
        ) == (44 + 5 * 76, 22, round(16 * 508 * 44 / 76))
        assert (
            cake_backend.router_partition(
                "nn_t", arch="sm_103a", sm_count=152, M=4096, N=256, K=16231
            )
            is None
        )
        assert cake_backend.router_partition(
            "nn_t", arch="sm_103a", sm_count=152, M=4096, N=256, K=20480
        ) == (44 + 4 * 76, 22, round(16 * 640 * 44 / 76))
        for arch in ("sm_100a", "sm_103a"):
            assert cake_backend.router_partition(
                "nn_t", arch=arch, sm_count=152, **wgrad5120
            ) == (304, 38, 20 * 508 // 2)
    assert cake_backend.router_partition(
        "nn_t", arch="sm_107a", sm_count=216, **wgrad5120
    ) == (540, 54, 20 * 508 // 2)
    for sms, pairs in ((212, 106), (208, 104)):
        assert cake_backend.router_partition(
            "nn_t", arch="sm_107a", sm_count=sms, **wgrad5120
        ) == (5 * pairs, pairs // 2, 20 * 508 // 2)
    for hidden in (4096, 6144):
        assert cake_backend.ROUTER_RULES[("sm_107a", "nn_t", hidden, 256)] is None
        # 16231 tokens: the deterministic plan (hidden 6144 on 212 SMs: "H0.5x4", below)
        for sms in (208, 212, 216):
            if (sms, hidden) == (212, 6144):
                continue
            assert (
                cake_backend.router_partition(
                    "nn_t", arch="sm_107a", sm_count=sms, M=hidden, N=256, K=16231
                )
                is None
            )
    assert cake_backend.router_partition(
        "nn_t", arch="sm_107a", sm_count=212, M=6144, N=256, K=16231
    ) == (5 * 106, 53, 24 * 508 // 2)
    # the forward of the parts with a table of their own: the deterministic plan
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
    # few tokens at hidden 5120: the "waves" partition on sm_100a and sm_103a (100 tokens, 4 K blocks: three waves of
    # the 76 CTA pairs, at most one unit per (tile, K block) step -> 160 units), no SIMT program
    for arch in ("sm_100a", "sm_103a"):
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=100
        ) == (160, 0, 0)
        assert (
            cake_backend.router_simt(
                "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=100
            )
            is None
        )
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


# The router windows of 152-SM parts that missed the accuracy rule (for some seed a relative Frobenius error against
# FP64 more than 1.02 times that of the plan the row takes without the window) and now run that plan, (arch, layout,
# hidden) -> ((first, last, spec), ...) (256 experts): the windows as they shipped before (of the sm_103a "waves"
# window, its token counts from 161 up), which tests of their programs force.
# fmt: off
FAILED_ROUTER_WINDOWS = {
    ("sm_100a", "nn_t", 4096): ((1, 224, "simt_n128m64s2"), (10240, 16384, "H1x5p44")),
    ("sm_100a", "nn_t", 5120): ((1, 256, "simt_n64m64s2"), (10240, 32768, "H0.5x3")),
    ("sm_100a", "nn_t", 6144): ((1, 256, "simt_n64m64s2"),),
    ("sm_103a", "nn_t", 4096): ((20480, 49152, "H1x4p44"),),
    ("sm_103a", "nn_t", 5120): ((161, 192, "waves"), (16172, 32768, "H0.5x3")),
    ("sm_103a", "nn_t", 6144): ((16172, 20480, "H1x5p44"),),
    ("sm_100a", "kn", 4096): ((1, 128, "simt_n64m32s4"),),
    ("sm_100a", "kn", 5120): ((1, 128, "simt_n64m32s4"),),
    ("sm_100a", "kn", 6144): ((1, 64, "simt_n64m32s4"),),
}
# fmt: on
# The windows of the parts with a router table of their own (ROUTER_RULES_BY_SM) that missed the accuracy rule and now
# run the plan the row takes without them (the deterministic program), (arch, SM count, layout, hidden) -> ((first,
# last, spec), ...) (256 experts): the SIMT windows of 212-SM sm_107a parts as they shipped before.
# fmt: off
FAILED_ROUTER_WINDOWS_BY_SM = {
    ("sm_107a", 212, "nn_t", 4096): ((1, 256, "simt_n128m64s2"),),
    ("sm_107a", 212, "nn_t", 5120): ((1, 256, "simt_n128m64s2"),),
    ("sm_107a", 212, "nn_t", 6144): ((1, 192, "simt_n128m64s2"),),
    ("sm_107a", 212, "kn", 4096): ((1, 64, "simt_n64m32s8"), (65, 128, "simt_n64m64s4")),
    ("sm_107a", 212, "kn", 5120): ((1, 64, "simt_n64m32s8"), (65, 128, "simt_n64m64s4")),
    ("sm_107a", 212, "kn", 6144): ((1, 64, "simt_n64m32s8"),),
}
# fmt: on


def _failed_router_window_rows() -> list:
    """``((arch, sm_count, layout, hidden), windows)`` of every window that missed the accuracy rule: those of
    :data:`FAILED_ROUTER_WINDOWS` on the measured SM count of their architecture (:data:`ROUTER_SM_COUNT`) and those
    of :data:`FAILED_ROUTER_WINDOWS_BY_SM`."""
    rows = [
        ((arch, cake_backend.ROUTER_SM_COUNT[arch], layout, hidden), windows)
        for (arch, layout, hidden), windows in FAILED_ROUTER_WINDOWS.items()
    ]
    return rows + list(FAILED_ROUTER_WINDOWS_BY_SM.items())


def _force_failed_router_windows(monkeypatch, arch, keep) -> int:
    """Set the windows of ``arch`` that missed the accuracy rule (:func:`_failed_router_window_rows`) whose spec
    ``keep`` accepts back into the router table of their SM count (:data:`ROUTER_RULES`, or the part's table of
    :data:`ROUTER_RULES_BY_SM`; before the row's shipped windows), so a test of their programs runs them on a part of
    the SM count they were measured on; returns how many windows it set."""
    n = 0
    for (rule_arch, sm_count, layout, hidden), windows in _failed_router_window_rows():
        forced = tuple(w for w in windows if keep(w[2]))
        if rule_arch != arch or not forced:
            continue
        table, _ = cake_backend.router_rules(arch, sm_count)
        key = (arch, layout, hidden, 256)
        shipped = table.get(key) or ()
        monkeypatch.setitem(table, key, forced + tuple(shipped))
        n += len(forced)
    cake_backend.clear_convenience_caches()
    return n


# the token windows of the router weight gradient (256 experts), (arch, hidden) -> ((first, last, spec), ...); every
# other token count runs the deterministic program.  A SIMT window comes first: where the table's windows do not
# hold, the token count takes the next window that holds it
ROUTER_WGRAD_WINDOWS = {
    ("sm_100a", 5120): ((1, 128, "waves"),),
    ("sm_103a", 5120): ((1, 160, "waves"),),
    ("sm_107a", 5120): ((16172, 16384, "H0.5x4"),),
}
# the parts with a router table of their own (ROUTER_RULES_BY_SM) and their weight-gradient windows,
# (arch, sm_count, hidden) -> ((first, last, spec), ...)
ROUTER_BY_SM_PARTS = {("sm_107a", 208), ("sm_107a", 212)}
ROUTER_WGRAD_WINDOWS_BY_SM = {
    ("sm_107a", 208, 5120): ((16172, 16384, "H0.5x4"),),
    ("sm_107a", 212, 5120): ((16172, 16384, "H0.5x4"),),
    ("sm_107a", 212, 6144): ((16172, 16384, "H0.5x4"),),
}


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_router_wgrad_token_windows(arch):
    """The weight gradient runs a partition only inside its token windows: at the first and the last token count of a
    window, not one token outside it, and around the few-token "waves" windows; a head/tail or SIMT window only on
    the SM count of its table (:data:`ROUTER_SM_COUNT` for :data:`ROUTER_RULES`, the part's own for
    :data:`ROUTER_RULES_BY_SM`; not one SM more or fewer), a "waves" window on every SM count where
    :func:`router_waves` gives a partition (below a SIMT window on the other SM counts).  A SIMT window runs the SIMT
    program of its tile (:func:`router_simt`), no partition.  The deterministic program at every other token count
    and SM count and on every hidden size without a window."""
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
                    want = None if units is None else (units, 0, 0)
                elif spec is None or not holds:
                    want = None
                else:
                    want = cake_backend.router_partition_of(
                        spec, m_pairs, 2, k_iters, sms
                    )
                assert got == want, (hidden, T, sms)
                if not str(spec).startswith("simt_"):
                    assert simt is None, (hidden, T, sms)
    if arch in ("sm_100a", "sm_103a"):
        # the smallest measured "waves" count (two K blocks: 80 units), no SIMT program
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=64
        ) == (80, 0, 0)
        assert (
            cake_backend.router_simt(
                "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=64
            )
            is None
        )


def test_router_windows_failing_the_accuracy_rule_run_the_deterministic_plan():
    """The router windows that missed the accuracy rule (:data:`FAILED_ROUTER_WINDOWS` on 152-SM parts,
    :data:`FAILED_ROUTER_WINDOWS_BY_SM` on 212-SM sm_107a parts) run the plan the row takes without them, at the first
    and the last token count of each window, one token inside each and a token count in the middle: the deterministic
    program -- no partition, no SIMT program, the planner keeps the row on the deterministic program -- except at
    hidden 5120 up to 128 tokens on sm_100a and up to 160 on sm_103a, which keep the row's "waves" window.  On 148 SMs
    (152-SM windows) and on 208 and 216 SMs (212-SM windows) every one of these token counts runs the deterministic
    program."""
    waves_last = {"sm_100a": 128, "sm_103a": 160}
    failed = []
    for (arch, sms, layout, hidden), windows in _failed_router_window_rows():
        if (arch, sms) in cake_backend.ROUTER_RULES_BY_SM:
            failed.append((arch, (208, sms, 216), layout, hidden, windows))
        else:
            assert sms == 152
            failed.append((arch, (148, sms), layout, hidden, windows))
    assert {f[0] for f in failed} == {"sm_100a", "sm_103a", "sm_107a"}
    checked = kept = 0
    for arch, sm_counts, layout, hidden, windows in failed:
        for first, last, _spec in windows:
            for T in sorted({first, first + 1, (first + last) // 2, last - 1, last}):
                M, N, K = _router_launch_dims(layout, hidden, 256, T)
                for sms in sm_counts:
                    kw = dict(arch=arch, sm_count=sms, M=M, N=N, K=K)
                    assert cake_backend.router_simt(layout, **kw) is None, (kw, layout)
                    spec = cake_backend._router_rule_spec(layout, **kw)
                    part = cake_backend.router_partition(layout, **kw)
                    waves_end = 0
                    if (layout, hidden) == ("nn_t", 5120):
                        waves_end = waves_last.get(arch, 0)
                    if waves_end < T:
                        assert spec is None and part is None, (kw, layout)
                        if layout == "nn_t":
                            # stride-0 views: the layout classes of G.t() and X
                            A = torch.empty(1, N).expand(K, N).t()
                            B, out = torch.empty(1, M).expand(K, M), torch.empty(N, M)
                        else:
                            A, B = torch.empty(M, K), torch.empty(K, N)
                            out = torch.empty(M, N)
                        plan, reason = cake_backend.plan_atomic_router(
                            A, B, out, arch=arch, sm_count=sms
                        )
                        assert plan is None, (kw, layout, reason)
                        checked += 1
                        continue
                    assert spec == "waves", (kw, layout)
                    k_iters = -(-T // 32)
                    units = cake_backend.router_waves(hidden // 256, 2, k_iters, sms)
                    if units is not None:
                        units = cake_backend.router_units_of(
                            units, hidden // 256, 2, k_iters, sms
                        )
                    assert part == (None if units is None else (units, 0, 0)), kw
                    assert (sms == 148) == (part is None), kw
                    kept += 1
    assert checked > 150 and kept


def test_router_per_sm_count_tables_decide_only_their_own_sm_count(monkeypatch):
    """A part whose SM count has a table in :data:`ROUTER_RULES_BY_SM` takes that table only (every row of
    :data:`ROUTER_RULES` of its architecture has an entry there, and rows measured on that SM count only may too; no
    ``"waves"`` window, closed head/tail or SIMT windows of the weight gradient, closed SIMT windows of the input
    gradient); one SM more or fewer runs the deterministic program, and the measured SM count of
    :data:`ROUTER_RULES` decides as before."""
    for (arch, sms), table in cake_backend.ROUTER_RULES_BY_SM.items():
        assert sms != cake_backend.ROUTER_SM_COUNT[arch]
        assert set(table) >= {k for k in cake_backend.ROUTER_RULES if k[0] == arch}
        for (key_arch, layout, _hidden, _experts), rule in table.items():
            assert key_arch == arch
            if rule is None:
                continue
            assert layout in ("nn_t", "kn") and isinstance(rule, tuple)
            for first, last, spec in rule:
                assert 1 <= first <= last <= 32768
                if layout == "kn":
                    assert cake_backend.router_simt_tile(spec) is not None
                    continue
                assert (
                    cake_backend.router_head_tail(spec) is not None
                    or cake_backend.router_simt_tile(spec) is not None
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
    (``router_fp32_dgrad_simt_m<tile_m>n<tile_n>s<slices>``); every other spec is none, and the forward has no SIMT
    program."""
    assert cake_backend.router_simt_tile("simt_n64m64s4") == (64, 64, 4)
    assert cake_backend.router_simt_tile("simt_n128m64s2") == (128, 64, 2)
    for spec in (
        None,
        3,
        "waves",
        "H0.5x3",
        "S2",
        "simt",
        "simt_n64m64",
        "xsimt_n64m64s4",
    ):
        assert cake_backend.router_simt_tile(spec) is None, spec
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
        # another SM count: no SIMT window (the deterministic program)
        for sms in (148, 151, 153, 160):
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
    # "waves" partition wherever router_waves gives one (as without the SIMT window)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, key, ((1, 256, "simt_n64m64s2"), (1, 128, "waves"))
    )
    m_pairs, waves_seen = hidden // 256, 0
    for sms in (148, 150, 152, 153, 160):
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
                waves_seen += 1
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


def _dgrad_simt_windows(forced) -> dict:
    """The input gradient's SIMT windows, ``(arch, SM count, hidden) -> ((first, last, tile), ...)`` (256 experts): as
    shipped none (the windows of 152-SM sm_100a and 212-SM sm_107a parts missed the accuracy rule); with ``forced``,
    those windows (:func:`_failed_router_window_rows`)."""
    if not forced:
        return {}
    return {
        (arch, sms, hidden): tuple(
            (lo, hi, cake_backend.router_simt_tile(spec)) for lo, hi, spec in windows
        )
        for (arch, sms, layout, hidden), windows in _failed_router_window_rows()
        if layout == "kn"
    }


def test_router_simt_windows_of_the_input_gradient(monkeypatch):
    """The input gradient (``kn``: ``M`` tokens, ``N`` hidden, ``K`` experts) takes the SIMT program of its window at
    every token count of the window, on the SM count of its table only; one token past the window, a caller's split
    count, another SM count or architecture, another expert count and a hidden size the tile does not divide run the
    deterministic program, and ``router_partition`` gives none (the input gradient has no stream-K program).  As
    shipped no row has a window (the windows of 152-SM sm_100a and 212-SM sm_107a parts missed the accuracy rule); the
    windows are checked forced too."""
    tokens = (1, 17, 32, 63, 64, 65, 96, 127, 128, 129, 192, 256, 1000, 16231)
    for forced in (False, True):
        if forced:
            for arch in ("sm_100a", "sm_107a"):
                _force_failed_router_windows(
                    monkeypatch,
                    arch,
                    lambda spec: cake_backend.router_simt_tile(spec) is not None,
                )
        shipped = _dgrad_simt_windows(forced)
        seen = 0
        for arch in cake_jit.ARCHES:
            for sms in ROUTER_SM_FACTS:
                for hidden in (4096, 5120, 6144):
                    windows = shipped.get((arch, sms, hidden), ())
                    for T in tokens:
                        want = next((t for lo, hi, t in windows if lo <= T <= hi), None)
                        kw = dict(arch=arch, sm_count=sms, M=T, N=hidden, K=256)
                        assert cake_backend.router_simt("kn", **kw) == want, (
                            arch,
                            sms,
                            hidden,
                            T,
                            forced,
                        )
                        assert cake_backend.router_simt("kn", splits=1, **kw) is None
                        assert cake_backend.router_partition("kn", **kw) is None
                        seen += want is not None
                        other = dict(kw, K=128)
                        assert cake_backend.router_simt("kn", **other) is None
        # forced: nine token counts from 1 to 128 at hidden 4096 and 5120, five from 1 to 64 at hidden 6144, on the
        # 152-SM sm_100a part and on the 212-SM sm_107a part
        assert seen == (2 * (9 + 9 + 5) if forced else 0)
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
    caller's split count and a token count past the window run the deterministic program.  The window is the 152-SM
    sm_100a one that missed the accuracy rule (:data:`FAILED_ROUTER_WINDOWS`), forced."""
    arch, hidden, E = "sm_100a", 5120, 256
    kw = dict(arch=arch, sm_count=152)
    key = (arch, "kn", hidden)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, (*key, E), FAILED_ROUTER_WINDOWS[key]
    )
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
    # another part (a 152-SM sm_103a part has no input-gradient window)
    plan, reason = cake_backend.plan_atomic_router(
        G, W, out, arch="sm_103a", sm_count=152
    )
    assert plan is None and "kn class runs the deterministic program" in reason


def test_router_sm_103a_head_tail_windows(monkeypatch):
    """The sm_103a weight gradient (256 experts) on 152 SMs runs the deterministic program at hidden 4096 and 6144 (the
    head/tail windows missed the accuracy rule).  With those windows forced (:data:`FAILED_ROUTER_WINDOWS`): hidden
    4096 runs "H1x4p44" (44 head and 304 tail units) at 20480 to 49152 tokens and hidden 6144 "H1x5p44" (44 head and
    380 tail units) at 16172 to 20480 tokens; the deterministic program one token outside, on 148-SM parts (no router
    table of their own) and one SM more or fewer than 152."""

    def part(sms, hidden, T):
        return cake_backend.router_partition(
            "nn_t", arch="sm_103a", sm_count=sms, M=hidden, N=256, K=T
        )

    for hidden in (4096, 6144):
        for T in (16172, 20480, 24576, 49152):
            assert part(152, hidden, T) is None, (hidden, T)
        key = ("sm_103a", "nn_t", hidden)
        monkeypatch.setitem(
            cake_backend.ROUTER_RULES, (*key, 256), FAILED_ROUTER_WINDOWS[key]
        )
    for T, k_iters in ((20480, 640), (24576, 768), (49152, 1536)):
        assert part(152, 4096, T) == (
            44 + 4 * 76,
            22,
            round(16 * k_iters * 44 / 76),
        )
    for T, k_iters in ((16172, 506), (16384, 512), (20480, 640)):
        assert part(152, 6144, T) == (
            44 + 5 * 76,
            22,
            round(24 * k_iters * 44 / 76),
        )
    for hidden, T in ((4096, 20479), (4096, 49153), (6144, 16171), (6144, 20481)):
        assert part(152, hidden, T) is None, (hidden, T)
    assert ("sm_103a", 148) not in cake_backend.ROUTER_RULES_BY_SM
    for sms in (148, 151, 153):
        for hidden, T in ((4096, 20480), (4096, 49152), (6144, 16172), (6144, 20480)):
            assert part(sms, hidden, T) is None, (sms, hidden, T)


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
    # a "waves" verdict at hidden 5120 (40 tiles, 129 K blocks: 3 waves of the 76 pairs)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_103a", "nn_t", 5120, 256), "waves"
    )
    X5 = torch.zeros(T, 5120)
    plan, _ = cake_backend.plan_atomic_router(G.t(), X5, torch.zeros(N, 5120), **kw)
    assert (plan.layout, plan.template, plan.swapped) == (
        "nn_t",
        "router_fp32_gemm_atomic_nn_t",
        True,
    )
    assert (plan.M, plan.N, plan.K, plan.k_iters) == (5120, N, T, 129)
    assert plan.units == 3 * 76 and plan.part_tiles and not plan.reproducible
    dgrad = (G, W, torch.zeros(T, K))
    assert cake_backend.plan_atomic_router(*dgrad, **kw)[0] is None
    assert plan.head_spans == 0 and plan.head_steps == 0


def test_k2_planner_plans_the_head_tail_router(monkeypatch):
    """A head/tail verdict (``"H0.5x4"`` on sm_107a, set here for hidden 4096, 5120 and 6144; sm_100a / sm_103a at
    hidden 5120) runs the head/tail program of the architecture: ``router_fp32_gemm_atomic_nn_t_ht_c3_nreg`` (on
    sm_103a ``router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg``) with the partition's ``head_spans`` /
    ``head_steps``, which the launch passes to the program; without that template registered the row runs the
    deterministic program."""
    arch, T, N = "sm_107a", 16231, 256
    kw = dict(arch=arch, sm_count=216)
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
    # sm_100a / sm_103a, hidden 5120 (the "H0.5x3" windows that missed the accuracy rule, forced): three tail units
    # per pair (76 head + 228 tail units on 152 SMs); sm_103a runs the head/tail program with the A parts in tensor
    # memory, and the deterministic program while this checkout does not register it
    templates = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES
    assert templates == {"sm_103a": ts}
    by_sm = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES_BY_SM
    assert by_sm == {("sm_107a", 212): ts}
    for gb, template in (
        ("sm_100a", "router_fp32_gemm_atomic_nn_t_ht_c3_nreg"),
        ("sm_103a", ts),
    ):
        assert cake_backend.router_head_tail_template(gb) == template
        key = (gb, "nn_t", 5120)
        monkeypatch.setitem(
            cake_backend.ROUTER_RULES, (*key, 256), FAILED_ROUTER_WINDOWS[key]
        )
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


def test_k2_head_tail_program_of_each_architecture_and_sm_count(monkeypatch):
    """With this checkout's registry and router tables, on every architecture and the SM counts of the measured parts,
    the weight gradient (256 experts, hidden 4096 / 5120 / 6144, token counts at and around the windows) runs the
    head/tail program with the A parts in tensor memory exactly in the sm_107a head/tail windows on 212 SMs (hidden
    5120 and 6144 at 16172 to 16384 tokens, the 212-SM table of ``ROUTER_RULES_BY_SM``,
    ``ROUTER_HEAD_TAIL_TEMPLATES_BY_SM``); every other head/tail plan runs ``router_fp32_gemm_atomic_nn_t_ht_c3_nreg``,
    and both carry the planner's partition.  sm_103a parts run no head/tail window (those of 152 SMs missed the
    accuracy rule); forced (:data:`FAILED_ROUTER_WINDOWS`), they run the program with the A parts in tensor memory.
    The program is registered for sm_103a and sm_107a."""
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
    # the head/tail windows of the tensor-memory program, (arch, sm_count, hidden) -> (first, last): sm_107a on 212
    # SMs (ROUTER_RULES_BY_SM)
    ts_windows = {
        ("sm_107a", 212, 5120): (16172, 16384),
        ("sm_107a", 212, 6144): (16172, 16384),
    }
    tokens = (8192, 10240, 16171, 16172, 16231, 16384, 16385, 20479, 20480, 24576)
    N, seen = 256, {}
    for arch in cake_jit.ARCHES:
        for sms in ROUTER_SM_FACTS:
            for hidden in (4096, 5120, 6144):
                out = torch.empty(N, hidden)
                for T in tokens + (24577, 32768, 32769, 49152, 49153):
                    # stride-0 views of the gradient [T, N] and the activations [T, hidden]: the layout classes of
                    # G.t() and X without their memory
                    G = torch.empty(1, N).expand(T, N)
                    X = torch.empty(1, hidden).expand(T, hidden)
                    plan, _ = cake_backend.plan_atomic_router(
                        G.t(), X, out, arch=arch, sm_count=sms
                    )
                    first, last = ts_windows.get((arch, sms, hidden), (1, 0))
                    in_window = first <= T <= last
                    if plan is None or not plan.head_spans:
                        assert not in_window, (arch, sms, hidden, T)
                        continue
                    assert plan.template == (ts if in_window else c3), (
                        arch,
                        sms,
                        hidden,
                        T,
                    )
                    assert (
                        plan.units,
                        plan.head_spans,
                        plan.head_steps,
                    ) == cake_backend.router_partition(
                        "nn_t", arch=arch, M=hidden, N=N, K=T, sm_count=sms
                    )
                    key = (arch, sms, plan.template)
                    seen[key] = seen.get(key, 0) + 1
    # sm_107a on 212 SMs: hidden 5120 and 6144 at 16172, 16231 and 16384 tokens
    assert seen.pop(("sm_107a", 212, ts)) == 3 + 3
    assert seen and all(template == c3 for _, _, template in seen)
    assert {(arch, sms) for arch, sms, _ in seen} == {
        ("sm_107a", 208),
        ("sm_107a", 216),
    }
    # the sm_103a head/tail windows that missed the accuracy rule, forced: the program with the A parts in tensor
    # memory at their first and last token counts
    head_tail = lambda spec: cake_backend.router_head_tail(spec) is not None  # noqa: E731
    assert _force_failed_router_windows(monkeypatch, "sm_103a", head_tail) == 3
    forced = 0
    for (arch, _layout, hidden), windows in FAILED_ROUTER_WINDOWS.items():
        if arch != "sm_103a":
            continue
        out = torch.empty(N, hidden)
        for first, last, spec in windows:
            if cake_backend.router_head_tail(spec) is None:
                continue
            for T in (first, last):
                G = torch.empty(1, N).expand(T, N)
                X = torch.empty(1, hidden).expand(T, hidden)
                plan, reason = cake_backend.plan_atomic_router(
                    G.t(), X, out, arch=arch, sm_count=152
                )
                assert plan is not None and plan.head_spans, (hidden, T, reason)
                assert plan.template == ts, (hidden, T)
                forced += 1
    assert forced == 6


#: The registry records (``cake_jit.MODULES``, as exported) of the SIMT router programs this checkout registered
#: before their windows missed the accuracy rule; CPU tests of a forced SIMT window bind them
#: (:func:`_register_former_simt`; nothing is loaded or launched).
# fmt: off
FORMER_SIMT_MODULES = {
    "cake_dense_projection_gemm_23cd6c7b07c9130bef18": {"template": "router_fp32_gemm_simt_n128m64s2", "kernel": "kernel_cake_dense_projection_gemm_23cd6c7b07c9130bef18", "library": "cake_dense_projection_gemm_23cd6c7b07c9130bef18", "ffi_entry": "run", "arg_plan": [["buffer", "A"], ["buffer", "B"], ["buffer", "out"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "lda"], ["parameter", "ldb"], ["parameter", "ldo"], ["launch", "enable_pdl"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_100a", "sm_107a"], "build": {"module": "tart.gemm_fp32.sm100_sm103_sm107.router_fp32_gemm_simt", "kwargs": {"tile_n": 128, "tile_m": 64, "slices": 2}}},
    "cake_dense_projection_gemm_5d34d9ca403abf798d9a": {"template": "router_fp32_dgrad_simt_m32n64s8", "kernel": "kernel_cake_dense_projection_gemm_5d34d9ca403abf798d9a", "library": "cake_dense_projection_gemm_5d34d9ca403abf798d9a", "ffi_entry": "run", "arg_plan": [["buffer", "A"], ["buffer", "B"], ["buffer", "out"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "lda"], ["parameter", "ldb"], ["parameter", "ldo"], ["launch", "enable_pdl"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_107a"], "build": {"module": "tart.gemm_fp32.sm100_sm103_sm107.router_fp32_dgrad_simt", "kwargs": {"tile_m": 32, "tile_n": 64, "slices": 8}}},
    "cake_dense_projection_gemm_7a259feabf02e3ac9e0f": {"template": "router_fp32_gemm_simt_n64m64s2", "kernel": "kernel_cake_dense_projection_gemm_7a259feabf02e3ac9e0f", "library": "cake_dense_projection_gemm_7a259feabf02e3ac9e0f", "ffi_entry": "run", "arg_plan": [["buffer", "A"], ["buffer", "B"], ["buffer", "out"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "lda"], ["parameter", "ldb"], ["parameter", "ldo"], ["launch", "enable_pdl"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_100a"], "build": {"module": "tart.gemm_fp32.sm100_sm103_sm107.router_fp32_gemm_simt", "kwargs": {"tile_n": 64, "tile_m": 64, "slices": 2}}},
    "cake_dense_projection_gemm_9f7c65acfbefcf9442c4": {"template": "router_fp32_dgrad_simt_m64n64s4", "kernel": "kernel_cake_dense_projection_gemm_9f7c65acfbefcf9442c4", "library": "cake_dense_projection_gemm_9f7c65acfbefcf9442c4", "ffi_entry": "run", "arg_plan": [["buffer", "A"], ["buffer", "B"], ["buffer", "out"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "lda"], ["parameter", "ldb"], ["parameter", "ldo"], ["launch", "enable_pdl"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_107a"], "build": {"module": "tart.gemm_fp32.sm100_sm103_sm107.router_fp32_dgrad_simt", "kwargs": {"tile_m": 64, "tile_n": 64, "slices": 4}}},
    "cake_dense_projection_gemm_f557f68a025cb6450dda": {"template": "router_fp32_dgrad_simt_m32n64s4", "kernel": "kernel_cake_dense_projection_gemm_f557f68a025cb6450dda", "library": "cake_dense_projection_gemm_f557f68a025cb6450dda", "ffi_entry": "run", "arg_plan": [["buffer", "A"], ["buffer", "B"], ["buffer", "out"], ["parameter", "M"], ["parameter", "N"], ["parameter", "K"], ["parameter", "lda"], ["parameter", "ldb"], ["parameter", "ldo"], ["launch", "enable_pdl"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]], "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]}, "arches": ["sm_100a"], "build": {"module": "tart.gemm_fp32.sm100_sm103_sm107.router_fp32_dgrad_simt", "kwargs": {"tile_m": 32, "tile_n": 64, "slices": 4}}},
}
# fmt: on


def _register_former_simt(monkeypatch, arch: str, *templates: str) -> None:
    """Register the former SIMT programs ``templates`` (:data:`FORMER_SIMT_MODULES`) on ``arch``: their records in
    ``cake_jit.MODULES`` and ``cake_jit.KERNELS`` (CPU binding tests of a forced SIMT window)."""
    by_template = {r["template"]: (n, r) for n, r in FORMER_SIMT_MODULES.items()}
    table = dict(cake_jit.KERNELS.get(arch, {}))
    for template in templates:
        name, record = by_template[template]
        assert arch in record["arches"], (arch, template)
        monkeypatch.setitem(cake_jit.MODULES, name, record)
        table[template] = name
    monkeypatch.setitem(cake_jit.KERNELS, arch, table)


def test_k2_values_feed_every_registered_router_program():
    """The keyword arguments of a K2 launch name every argument of every registered stream-K router program's
    argument plan (the head/tail program's ``head_spans`` / ``head_steps`` included); those of a SIMT launch every
    argument of every registered SIMT program's plan (the weight gradient's and the input gradient's; while this
    checkout registers none, those of :data:`FORMER_SIMT_MODULES`, the SIMT programs it registered before), the
    operands' row strides and the launch flag before the grid, 0 whatever the call's PDL flag (a SIMT program never
    launches as a programmatic dependent launch)."""
    out = torch.zeros(256, 5120)
    plan = cake_backend.AtomicRouterPlan(
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
    simt_records = [
        (name, cake_jit.arg_plan_of(record))
        for name, record in cake_jit.MODULES.items()
        if record["template"].startswith(simt_prefix)
    ] or [
        (name, cake_jit.arg_plan_of(record))
        for name, record in FORMER_SIMT_MODULES.items()
        if record["template"].startswith(simt_prefix)
    ]
    assert simt_records
    for name, plan_args in simt_records:
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
    dgrad_records = [
        (name, cake_jit.arg_plan_of(record))
        for name, record in cake_jit.MODULES.items()
        if record["template"].startswith(dgrad_prefix)
    ] or [
        (name, cake_jit.arg_plan_of(record))
        for name, record in FORMER_SIMT_MODULES.items()
        if record["template"].startswith(dgrad_prefix)
    ]
    assert dgrad_records
    for name, plan_args in dgrad_records:
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
    # the stream-K router programs (the partial sum program has its own plan: test_partials_sum_serves_...)
    names = [
        n
        for n, r in cake_jit.MODULES.items()
        if r["template"].startswith("router_")
        and not r["template"].startswith((simt_prefix, dgrad_prefix))
        and r["template"] != cake_backend.PARTIALS_SUM_TEMPLATE
    ]
    for name in names:
        plan_args = cake_jit.arg_plan_of(cake_jit.MODULES[name])
        arguments = cake_backend._order_atomic_arguments(
            name, plan_args, values, plan.grid
        )
        assert len(arguments) == len(plan_args), name
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
    # the input gradient's SIMT window: a 152-SM sm_100a part (the window that missed the accuracy rule, forced); the
    # others' stream-K program with two splits
    key = ("sm_100a", "kn", H)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, (*key, 256), FAILED_ROUTER_WINDOWS[key]
    )
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
            plan, A_eff, B_eff = det_backend.plan_router_fp32_det_gemm(A, B, out)
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
    assert plan == det_backend.plan_router_fp32_det_gemm(A, B, out, splits=splits)[0]
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
    conversion as the identity; returns that entry."""
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
    for cache in ("_K1_ATOMIC_CACHE", "_K1_DET_CACHE"):
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
    """The K1 entry points with ``deterministic=False`` plan once per key and launch the bound program after: the
    forward and the input gradient in ``_K1_DET_CACHE`` (keyed by their own tensors), the weight gradient in
    ``_K1_ATOMIC_CACHE`` -- a row without a verdict bound to the deterministic package's plan of
    ``projection_det_wgrad`` (its orientation; another ``sk`` is another key and plan), a row with one to this
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
            cake_backend.projection_forward(X, W, out_f, deterministic=False) is out_f
        )
        assert cake_backend.projection_dgrad(G, W, out_d, deterministic=False) is out_d
        assert cake_backend.projection_wgrad(G, X, out_w, deterministic=False) is out_w
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
        cake_backend.projection_forward(
            X, W, out_f, deterministic=False, enable_pdl=False
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
    cake_backend.projection_wgrad(G, X, out_w, sk=False, deterministic=False)
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
    cake_backend.projection_wgrad(G, X, out_w, deterministic=False)
    ((_, atomic),) = cake_backend._K1_ATOMIC_CACHE.items()
    assert atomic.plan is not None and atomic.plan.splits == 2 and atomic.k1 is not None
    assert atomic.det_plan is None and atomic.arena is not None


def test_k1_entry_points_go_from_their_key_to_the_bound_launch(monkeypatch):
    """Once a key is bound, the K1 entry points with ``deterministic=False`` launch it straight from the key
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
            cake_backend.projection_forward(X, W, out_f, deterministic=False) is out_f
        )
        assert cake_backend.projection_dgrad(G, W, out_d, deterministic=False) is out_d
        assert cake_backend.projection_wgrad(G, X, out_w, deterministic=False) is out_w

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
    assert cake_backend.projection_forward(X, W, other, deterministic=False) is other
    (args, _) = entry_fn.calls[-1]
    pointers = {a.data_ptr() for a in args if isinstance(a, torch.Tensor)}
    assert other.data_ptr() in pointers and out_f.data_ptr() not in pointers
    with pytest.raises(AssertionError, match="planning path"):
        cake_backend.projection_forward(X[:999], W, out_f[:999], deterministic=False)
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
    ``K1_SM107A_WIDE_FP32_ROWS`` binds its one-split program one token below ``K1_SM107A_FP32_BUDGET_TOKENS`` and
    the deterministic package's plan at that token count and at 65536 tokens, after the smaller token count's entry
    is cached."""
    arch, T0 = "sm_107a", K1_SM107A_FP32_BUDGET_TOKENS
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    facts = det_backend.DeviceFacts(
        index=0, arch=arch, sm_count=216, l2_bytes=126 << 20
    )
    monkeypatch.setattr(det_backend, "device_facts", lambda device=None: facts)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    bf16 = torch.bfloat16
    for (d_in, d_out), choice in K1_SM107A_WIDE_FP32_ROWS.items():
        cake_backend._K1_ATOMIC_CACHE.clear()
        entry_fn.calls.clear()
        out = torch.empty(d_out, d_in)
        for T in (T0 - 1, T0, 65536):
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
                    cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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


@pytest.mark.parametrize("arch, sm_count", [("sm_100a", 152), ("sm_103a", 150)])
def test_k1_odd_block_exclusions_bind_per_token_count(monkeypatch, arch, sm_count):
    """The convenience path plans each token count of a K1 call at the part's SM count, so the entries of
    :data:`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` hold there too: for each FP32 row of the architecture in
    ``K1_ODD_BLOCK_EXCLUDED_TOKENS`` with an entry on the part, the weight gradient binds the row's two-split program
    one token below the first token count of its range (65537, an odd block count) and one token past the end of that
    token count's block (an even block count), and the deterministic package's plan at that first token count and at the
    last token count of its block (an odd block count), each token count its own cache entry.  (On 152-SM sm_103a parts
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
        assert lo == 65538 and _k1_blocks(lo) % 2 == 1
        cases = ((lo - 1, rule), (lo, None), (last, None), (last + 1, rule))
        for T, choice in cases:
            G, X = torch.empty(T, d_out, dtype=bf16), torch.empty(T, d_in, dtype=bf16)
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
    the 216-SM override of ``4096 -> 1024`` at 16384 tokens, the one-split program of ``5120 -> 1024`` at 32768 tokens
    on 216 SMs), each token count its own cache entry."""
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
                    cases.append((T, (block_n, 1, chains)))
        misses = K1_CHAIN_MISSES[sm_count].get((d_in, d_out), ())
        for T in misses:
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
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
    not bound here: its entry stays unbound and ``dense_projection_gemm`` makes the deterministic package's call, with
    ``enable_pdl`` and ``weights_static`` (part of the key); a batched forward of more tokens is bound."""
    arch = "sm_103a"
    entry_fn = _cpu_k1_launches(monkeypatch, arch)
    monkeypatch.setattr(cake_backend, "_k1_index", lambda *views: 0)
    det = _patch_det(monkeypatch, "dense_projection_gemm")
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
            result = cake_backend.dense_projection_gemm(
                A,
                B,
                out,
                deterministic=False,
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
        cake_backend.dense_projection_gemm(
            A, B, out, deterministic=False, enable_pdl=True
        )
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
                    assert part == cake_backend.router_partition_of(
                        spec, m_tiles // 2, n_tiles, k_iters, sm_count
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
    X, W = torch.empty(64, 5120), torch.empty(256, 5120)
    out = torch.empty(64, 256)
    without_verdict = cake_backend._k2_row_without_verdict
    assert without_verdict(X, W.t(), out, arch, sm_count)
    # bindings the planner refuses or classifies otherwise stay with it
    assert not without_verdict(X.bfloat16(), W.t(), out, arch, sm_count)
    assert not without_verdict(X, W.t(), out.double(), arch, sm_count)
    assert not without_verdict(X[None], W.t(), out, arch, sm_count)
    assert not without_verdict(X, W.t(), out[:, :128], arch, sm_count)
    assert not without_verdict(X[:0], W.t(), out[:0], arch, sm_count)
    assert not without_verdict(X, W.t()[:4096], out, arch, sm_count)
    # the kn class (the input gradient): no SIMT window on sm_103a or on 152-SM sm_100a parts (their windows missed
    # the accuracy rule); a window set for it is planned
    g, w, dx = torch.empty(64, 256), torch.empty(256, 5120), torch.empty(64, 5120)
    assert without_verdict(g, w, dx, arch, 152)
    assert cake_backend.plan_atomic_router(g, w, dx, arch=arch, sm_count=152)[0] is None
    assert without_verdict(g, w, dx, "sm_100a", 152)
    key = ("sm_100a", "kn", 5120)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, (*key, 256), FAILED_ROUTER_WINDOWS[key]
    )
    assert not without_verdict(g, w, dx, "sm_100a", 152)
    assert not without_verdict(
        torch.zeros(1).expand(64, 5120), W.t(), out, arch, sm_count
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
    change: no forward verdict ships; a forward verdict set in any table, and its removal, are seen at the next call;
    a table rebound in place of one is seen, and one that does not count its changes (a plain ``dict``) is read on
    every call; ``clear_convenience_caches`` drops the memo."""
    cake_backend.clear_convenience_caches()
    memo = cake_backend._router_memo()
    assert memo.tracked and not memo.forward_verdicts and not memo.without_verdict
    assert cake_backend._router_memo() is memo
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
    cake_backend.clear_convenience_caches()
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
    cake_backend.clear_convenience_caches()
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
    # the 212-SM hidden-6144 window (head/tail, 16172 to 16384 tokens)
    shipped = own[key]
    monkeypatch.setitem(own, key, ((1, 1024, "pairs"),))
    assert not runs(geometry, 1) and runs(geometry, 0) and runs(geometry, 2)
    own[key] = shipped
    assert runs(geometry, 1)
    assert check() == kept
    # a full memo starts over (from an empty memo: every check is a miss)
    monkeypatch.setattr(cake_backend, "_ROUTER_MEMO_SIZE", 5)
    cake_backend.clear_convenience_caches()
    assert check() == kept
    assert 0 < len(cake_backend._router_memo().without_verdict) <= 5


def test_k2_memo_under_concurrent_table_changes(monkeypatch):
    """Threads checking router calls while another thread sets and removes a verdict: no call fails, and once the
    tables stop changing every conclusion is the planner's for the final tables (a conclusion drawn while a table
    changed is not kept past that change)."""
    _fake_device_facts(monkeypatch, {0: ("sm_103a", 152)})
    cake_backend.clear_convenience_caches()
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
# ROUTER_RULES_BY_SM on 208-SM and 212-SM R200 parts
ROUTER_TABLE_PARTS = (
    ("sm_100a", 152),
    ("sm_103a", 152),
    ("sm_107a", 208),
    ("sm_107a", 212),
    ("sm_107a", 216),
)


def test_router_rule_spec_reads_the_table_of_the_sm_count(monkeypatch):
    """The convenience path's no-verdict check reads the router table of the device's SM count (``router_rules``):
    ``ROUTER_RULES`` on 152 SMs (sm_100a, sm_103a) and 216 SMs (sm_107a), the part's own table of
    ``ROUTER_RULES_BY_SM`` on 208 and 212 SMs (sm_107a).  At every row of that table and the edges of its windows the
    spec is the table's, ``None`` exactly where the device's planner keeps the row on the deterministic program for
    want of a verdict; a verdict in only one table is seen on exactly its SM count."""
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
    # the sm_107a hidden-5120 weight gradient: "H0.5x4" at 16172 to 16384 tokens on 208, 212 and 216 SMs; hidden
    # 6144: on 212 SMs only
    for sms in (208, 212, 216):
        for T, want in ((16171, None), (16172, "H0.5x4"), (16384, "H0.5x4")):
            assert (
                cake_backend._router_rule_spec(
                    "nn_t", arch="sm_107a", sm_count=sms, M=5120, N=256, K=T
                )
                == want
            )
        for T in (16171, 16172, 16231, 16384, 16385):
            want = "H0.5x4" if sms == 212 and 16172 <= T <= 16384 else None
            assert (
                cake_backend._router_rule_spec(
                    "nn_t", arch="sm_107a", sm_count=sms, M=6144, N=256, K=T
                )
                == want
            ), (sms, T)
    # a verdict in the 212-SM table only (as shipped), then in ROUTER_RULES only (216 SMs): the no-verdict check of a
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


def _same_arguments(got: list, want: list) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if isinstance(w, torch.Tensor):
            assert isinstance(g, torch.Tensor) and _geometry(g) == _geometry(w), i
        else:
            assert type(g) is type(w) and g == w, (i, g, w)


@pytest.mark.parametrize(
    "case",
    ["wgrad_waves", "wgrad_head_tail", "fwd_two_splits", "padded_out", "torch_chain"],
)
def test_k2_bound_arguments_are_the_ordered_arguments(monkeypatch, case):
    """A cached K2 launch orders its arguments once (``_k2_binding``) and binds the arena's BF16x3 stack once per arena
    growth: with a call's views in place its kernel and fused-split argument lists are the lists the launch orders
    per call from ``_k2_values`` / ``_split_values`` (the same tensors' geometry, the same values in the same
    positions), ``out`` stands for both of its arguments, and the fill geometry is ``_out_fill``'s.  A grown arena is
    carved again."""
    monkeypatch.setattr(cake_backend.tvm_ffi, "from_dlpack", lambda t, **_: t)
    monkeypatch.setattr(cake_backend, "_CU_DRIVER", False)
    arch, sm_count, T, H, splits = "sm_103a", 152, 64, 5120, None
    if case == "wgrad_head_tail":
        # the "H0.5x3" window that missed the accuracy rule, forced
        T = 16172
        key = (arch, "nn_t", H)
        monkeypatch.setitem(
            cake_backend.ROUTER_RULES, (*key, 256), FAILED_ROUTER_WINDOWS[key]
        )
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
# this package's program -- else the deterministic program through this package's binding); the head/tail case runs
# with the "H0.5x3" window that missed the accuracy rule forced (FAILED_ROUTER_WINDOWS)
# fmt: off
ROUTER_ENTRY_CASES = {
    "fwd": ("fwd", 64, 5120, None, False, False, False),
    "fwd_two_splits": ("fwd", 64, 5120, 2, False, False, True),
    "fwd_padded_out": ("fwd", 64, 5120, None, True, False, False),
    "dgrad": ("dgrad", 64, 5120, None, False, False, False),
    "dgrad_padded_out": ("dgrad", 64, 5120, None, True, False, False),
    "wgrad_waves": ("wgrad", 64, 5120, None, False, False, True),
    "wgrad_head_tail": ("wgrad", 16172, 5120, None, False, False, True),
    "wgrad_no_verdict": ("wgrad", 1000, 6144, None, False, False, False),
    "wgrad_padded_out": ("wgrad", 64, 5120, 2, True, False, True),
    "wgrad_torch_chain": ("wgrad", 64, 5120, None, False, True, True),
}
# fmt: on


def _router_entry_call(case):
    """``(call, out)``: a router entry point's ``deterministic=False`` call of ``case`` (:data:`ROUTER_ENTRY_CASES`)
    on fresh CPU tensors, and its output."""
    op, T, H, splits, pad_out, pad_g, _ = ROUTER_ENTRY_CASES[case]
    X, W = torch.empty(T, H), torch.empty(256, H)
    G = torch.empty(T, 256 + 16)[:, :256] if pad_g else torch.empty(T, 256)
    rows, cols = {"fwd": (T, 256), "dgrad": (T, H), "wgrad": (256, H)}[op]
    out = torch.empty(rows, cols + 64)[:, :cols] if pad_out else torch.empty(rows, cols)
    kw = (
        dict(deterministic=False)
        if op == "dgrad"
        else dict(deterministic=False, splits=splits)
    )
    call = {
        "fwd": lambda: cake_backend.router_forward(X, W, out, **kw),
        "dgrad": lambda: cake_backend.router_dgrad(G, W, out, **kw),
        "wgrad": lambda: cake_backend.router_wgrad(G, X, out, **kw),
    }[op]
    return call, out


@pytest.mark.parametrize("case", sorted(ROUTER_ENTRY_CASES))
def test_router_entry_points_launch_their_paths_launches(monkeypatch, case):
    """With ``deterministic=False`` a router entry point's call goes through ``_K2_ENTRY_CACHE``: planned once per key
    of its own tensors into the cached launch its own path takes (the same ``_K2_SPLITK_CACHE`` / ``_K2_ATOMIC_CACHE``
    entry, under the same key), it issues that path's launches -- the same programs with the same arguments, the
    same split and zero fill -- and the next calls go from the key to the bound launch (no geometry, layout, key of
    the launch or planning)."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch)
    split = Recorder()  # the torch chain's split
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    op, *_, atomic = ROUTER_ENTRY_CASES[case]
    if case == "wgrad_head_tail":
        key = ("sm_103a", "nn_t", 5120)
        monkeypatch.setitem(
            cake_backend.ROUTER_RULES, (*key, 256), FAILED_ROUTER_WINDOWS[key]
        )
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


@pytest.mark.parametrize("T", [1, 17, 64, 128, 129])
def test_router_dgrad_entry_binds_the_simt_program(monkeypatch, T):
    """With ``deterministic=False`` the input gradient's call in a SIMT window of a 152-SM sm_100a part (the window
    that missed the accuracy rule, forced) binds the SIMT program's launch in ``_K2_ENTRY_CACHE`` (planned once into
    ``_K2_ATOMIC_CACHE``): one launch per call of the program with the entry point's tensors -- ``G[None]``, ``W``
    and ``out`` read in place --, the launch's dimensions and row strides, no PDL flag and the grid of ``out``'s
    tiles; no split, no fill.  One token past the window the call binds the deterministic program (one split)
    through this package's binding."""
    entry_fn, fill = _cpu_k2_launches(monkeypatch, "sm_100a")
    # this checkout registers no SIMT program (their windows missed the accuracy rule): register the forced one
    _register_former_simt(monkeypatch, "sm_100a", "router_fp32_dgrad_simt_m32n64s4")
    split = Recorder()
    monkeypatch.setattr(det_backend, "split_fp32_to_bf16x3_det_", split)
    H, E = 5120, 256
    key = ("sm_100a", "kn", H)
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, (*key, E), FAILED_ROUTER_WINDOWS[key]
    )
    G, W = torch.empty(T, E), torch.empty(E, H)
    out = torch.empty(T, H + 64)[:, :H]
    for _ in range(3):
        assert cake_backend.router_dgrad(G, W, out, deterministic=False) is out
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


def test_router_entry_cache_keys_and_routes_again(monkeypatch):
    """``_K2_ENTRY_CACHE`` keys a call by the device, the raw stream, the operation, the split count, the PDL flag and
    the entry point's tensors, and routes a key again after a router table changes (a forward verdict set at run time
    runs this package's program, its removal the deterministic one) or a ``ROUTES`` value does; a call whose tensors
    are not on one CUDA device, or made while a router table is a plain ``dict``, takes the entry point's own path.
    ``clear_convenience_caches`` drops the entries."""
    arch, sm_count = "sm_103a", 152
    X, W, out = torch.empty(64, 5120), torch.empty(256, 5120), torch.empty(64, 256)
    assert cake_backend._k2_entry("fwd", X, W, out, None, None) is None  # CPU tensors
    entry_fn, _ = _cpu_k2_launches(monkeypatch, arch)
    cache = cake_backend._K2_ENTRY_CACHE

    def fwd(**kw):
        n = len(entry_fn.calls)
        assert cake_backend.router_forward(X, W, out, deterministic=False, **kw) is out
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
    assert table.get(key) is None  # no verdict on any architecture
    before = fwd()
    assert not before.atomic and before.launch is not None
    monkeypatch.setitem(table, key, "forward")
    after = fwd()
    assert after.atomic and after.memo is not before.memo and after.route == "atomic"
    assert after.launch.plan is not None and after.launch.k2 is not None
    table[key] = None
    assert not fwd().atomic
    # a ROUTES value changed at run time
    X, G, out = torch.empty(64, 5120), torch.empty(64, 256), torch.empty(256, 5120)

    def wgrad():
        n = len(entry_fn.calls)
        assert cake_backend.router_wgrad(G, X, out, deterministic=False) is out
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
        assert cake_backend.router_wgrad(G, X, out, deterministic=False) is out
        assert len(entry_fn.calls) > n and len(cache) == size
    cake_backend.clear_convenience_caches()
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
@pytest.mark.parametrize("path", ["entry_point", "router_fp32_gemm"])
@pytest.mark.parametrize("op", sorted(ROUTER_VERDICT_CHANGES))
def test_router_verdict_changed_at_run_time_plans_the_launch_again(
    monkeypatch, op, path, tracked
):
    """A router verdict changed between two calls of the same tensors: the next call runs the launch the planner gives
    under the new verdict, not the cached launch of the old one -- through the router entry point (its
    ``_K2_ENTRY_CACHE`` entry and the ``_K2_ATOMIC_CACHE`` launch under it) and through ``router_fp32_gemm``
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
    # this checkout registers no SIMT program (their windows missed the accuracy rule): register the verdicts' ones
    _register_former_simt(
        monkeypatch,
        arch,
        *{
            cake_backend.router_simt_template(cake_backend.router_simt_tile(s), key[1])
            for verdict in (first, second)
            if isinstance(verdict, tuple)
            for _, _, s in verdict
            if cake_backend.router_simt_tile(s) is not None
        },
    )
    X, W, G = torch.empty(T, H), torch.empty(256, H), torch.empty(T, 256)
    A, B, out = {
        "fwd": (X, W.t(), torch.empty(T, 256)),
        "dgrad": (G, W, torch.empty(T, H)),
        "wgrad": (G.t(), X, torch.empty(256, H)),
    }[op]
    entry_point = {
        "fwd": lambda: cake_backend.router_forward(X, W, out, deterministic=False),
        "dgrad": lambda: cake_backend.router_dgrad(G, W, out, deterministic=False),
        "wgrad": lambda: cake_backend.router_wgrad(G, X, out, deterministic=False),
    }[op]

    def call():
        """The cached launch a call ran, after checking that it launched."""
        n = len(entry_fn.calls)
        if path == "router_fp32_gemm":
            assert cake_backend.router_fp32_gemm(A, B, out, deterministic=False) is out
        else:
            assert entry_point() is out
        assert len(entry_fn.calls) > n
        ((_, launch),) = cake_backend._K2_ATOMIC_CACHE.items()
        if path == "entry_point" and tracked:
            ((_, entry),) = cake_backend._K2_ENTRY_CACHE.items()
            assert entry.atomic and entry.launch is launch
        else:  # router_fp32_gemm, or an entry point while a router table is a plain dict
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
    if path == "router_fp32_gemm" and op != "dgrad":

        def with_splits():
            assert (
                cake_backend.router_fp32_gemm(A, B, out, splits=2, deterministic=False)
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
    G, W, out = torch.empty(64, 256), torch.empty(256, 5120), torch.empty(64, 5120)
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
        # no input-gradient verdict on a 152-SM sm_103a part: the deterministic program
        assert entry.memo is memo and entry.layout == "k2_kn" and not entry.atomic
    monkeypatch.setattr(
        cake_backend, "_plan_splitk_k2", lambda *args: plan(dgrad_flags, None)
    )
    cake_backend._K2_SPLITK_CACHE.clear()
    path = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", path)
    assert cake_backend.router_dgrad(G, W, out, deterministic=False) is path.result
    ((key, entry),) = cake_backend._K2_ENTRY_CACHE.items()
    assert entry.launch is None and key[2] == ("dgrad", None, 1)
    monkeypatch.setattr(
        cake_backend, "_plan_k2_entry", lambda *args: pytest.fail("planned again")
    )
    assert cake_backend.router_dgrad(G, W, out, deterministic=False) is path.result
    assert len(path.calls) == 2


def test_router_entry_cache_holds_its_arenas(monkeypatch):
    """An entry of ``_K2_ENTRY_CACHE`` holds a reference to its launch's arena, as the launch's own entry does: an
    arena's reference count is the number of cached entries of every cache that use it, eviction (the cache bound)
    drops them, and an arena no entry uses any more is released."""
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
        assert cake_backend.router_dgrad(G, W, out, deterministic=False) is out
        arenas = list(cake_backend._ATOMIC_ARENAS.values())
        for arena in arenas:
            users = sum(e.arena is arena for c in caches for e in c.values())
            assert arena.refs == users > 0
        assert all(
            any(e.arena is a for a in arenas) for c in caches for e in c.values()
        )
        return {a.key: a.refs for a in arenas}

    for T in (64, 128, 192, 256, 320):
        refs = dgrad(T)
    assert len(caches[0]) == len(caches[1]) == 3 and refs == {(0, 0): 6}
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 7)
    assert dgrad(64) == {(0, 0): 4, (0, 7): 2}  # an arena of the stream's own
    monkeypatch.setattr(det_backend, "_current_raw_stream", lambda index: 0)
    for T in (64, 128, 192):
        refs = dgrad(T)
    assert refs == {(0, 0): 6}  # evicted from both caches: released
    cake_backend.clear_convenience_caches()
    assert not cake_backend._ATOMIC_ARENAS and not any(caches)


def test_deterministic_backend_follows_the_binding_and_sys_modules(monkeypatch):
    """``deterministic_backend`` returns the module an import of the bound package's ``cake_backend`` returns: a
    rebinding and a replaced ``sys.modules`` entry are seen on the next call."""
    assert cake_backend.deterministic_backend() is det_backend
    fake_pkg, fake = (
        ModuleType("vendored_det_pkg2"),
        ModuleType("vendored_det_pkg2.cake_backend"),
    )
    for name in cake_backend.DETERMINISTIC_NAMES.values():
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
    for name in cake_backend.DETERMINISTIC_NAMES.values():
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
    det = _patch_det(monkeypatch, "plan_router_fp32_gemm")
    monkeypatch.setattr(
        cake_backend,
        "_device_and_arch",
        lambda A, B, o: (torch.device("cpu"), arch, 152),
    )
    assert (
        cake_backend.plan_router_fp32_gemm(X, W.t(), out, deterministic=False)
        is det.result
    )
    ((args, kwargs),) = det.calls
    assert args[0] is X and args[1].data_ptr() == W.data_ptr() and args[2] is out
    assert kwargs == {"splits": None}


# ---------------------------------------------------------------------------
# PyTorch's deterministic-algorithms mode (CPU)
# ---------------------------------------------------------------------------

# fmt: off
ATOMIC_MODE_CALLS = {
    "dense_projection_gemm": lambda: cake_backend.dense_projection_gemm(*_k1_views("nn"), deterministic=False),
    "projection_forward": lambda: cake_backend.projection_forward("X", "W", deterministic=False),
    "projection_wgrad": lambda: cake_backend.projection_wgrad("G", "X", "out", deterministic=False),
    "router_forward": lambda: cake_backend.router_forward("X", "W", deterministic=False),
    "router_dgrad": lambda: cake_backend.router_dgrad("G", "W", deterministic=False),
    "router_wgrad": lambda: cake_backend.router_wgrad("G", "X", deterministic=False),
    "split_fp32_to_bf16x3_": lambda: cake_backend.split_fp32_to_bf16x3_("s", "p", "r", deterministic=False),
}
# fmt: on


@pytest.mark.parametrize("name", sorted(ATOMIC_MODE_CALLS))
def test_explicit_false_under_torch_deterministic_mode_raises(
    monkeypatch, torch_determinism, name
):
    recorder = _patch_det(monkeypatch, name)
    monkeypatch.setattr(cake_backend, "_atomic_k1", Recorder())
    monkeypatch.setattr(cake_backend, "_atomic_k2", Recorder())
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        ATOMIC_MODE_CALLS[name]()
    assert not recorder.calls


@pytest.mark.parametrize("name", sorted(ATOMIC_MODE_CALLS))
def test_explicit_false_under_warn_only_warns_and_runs_the_deterministic_path(
    monkeypatch, torch_determinism, name
):
    recorder = _patch_det(monkeypatch, name)
    atomic1, atomic2 = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k1", atomic1)
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running the deterministic path"):
        assert ATOMIC_MODE_CALLS[name]() is recorder.result
    assert len(recorder.calls) == 1 and not atomic1.calls and not atomic2.calls


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
    and their FP32 programs of several accumulation chains (``..._c<chains>``, :data:`K1_CHAINS_CHOICES`), the two
    router classes, the weight gradient's head/tail programs (of the architecture and of its SM counts), the
    SIMT programs the windows of the weight gradient and the input gradient name (in :data:`ROUTER_RULES` and the
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
    by_sm = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES_BY_SM
    rule_tables = (
        cake_backend.ROUTER_RULES,
        *cake_backend.ROUTER_RULES_BY_SM.values(),
    )
    for arch, table in cake_jit.KERNELS.items():
        heads = {cake_backend.router_head_tail_template(arch)} | {
            template for (sm_arch, _), template in by_sm.items() if sm_arch == arch
        }
        simt = {
            cake_backend.router_simt_template(
                cake_backend.router_simt_tile(spec), key[1]
            )
            for rules in rule_tables
            for key, rule in rules.items()
            if key[0] == arch and isinstance(rule, tuple)
            for _, _, spec in rule
            if cake_backend.router_simt_tile(spec) is not None
        }
        assert simt <= set(table), (arch, sorted(simt - set(table)))
        assert set(table) <= wanted | heads | simt, sorted(set(table) - wanted)


def test_no_det_token_in_generated_names_and_no_negated_token_in_the_package():
    root = Path(cake_jit.__file__).parent
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and "aot" not in p.parts
    ]
    for path in files:
        rel = path.relative_to(root).as_posix()
        assert not DET_TOKEN.search(path.name), rel
        assert not NEGATED_TOKEN.search(path.read_text(errors="replace")), rel
    generated = [*cake_jit.MODULES, *(r["kernel"] for r in cake_jit.MODULES.values())]
    generated += [t for table in cake_jit.KERNELS.values() for t in table]
    assert not [n for n in generated if DET_TOKEN.search(n)]
    manifest = root / "build.json"
    if manifest.exists():
        document = json.loads(manifest.read_text())
        # the in-tree import path: a vendored copy keeps its manifest
        assert document["package"] == "flashinfer.experimental.dense_projection_gemm"
        for entry in document["libraries"]:
            assert not DET_TOKEN.search(entry["name"])
            assert all(
                s.startswith("csrc/cake_dense_projection_gemm/")
                or s == "csrc/cake_ffi_checks.h"
                or s.startswith("csrc/cake_device_helpers/")
                for s in entry["sources"]
            )
    for name in ENTRY_POINTS:
        assert not DET_TOKEN.search(name)


def test_select_module_and_the_loader_forward_deterministic_names(monkeypatch):
    for arch, table in det_jit.KERNELS.items():
        template = sorted(table)[0]
        assert cake_jit.select_module(arch, template) == det_jit.select_module(
            arch, template
        )
    with pytest.raises(NotImplementedError, match="not registered"):
        cake_jit.select_module("sm_100a", "dense_proj_gemm_atomic_nn_unregistered")
    loader = Recorder()
    monkeypatch.setattr(det_jit, "load_cake_dense_projection_det_gemm_module", loader)
    name = sorted(det_jit.MODULES)[0]
    assert (
        cake_jit.load_cake_dense_projection_gemm_module(name, "sm_100a")
        is loader.result
    )
    assert loader.calls == [((name, "sm_100a"), {})]
    bind = Recorder()
    monkeypatch.setattr(det_backend, "bind_launch", bind)
    assert cake_backend.bind_launch(name, {"M": 1}, (2, 1, 1), "sm_100a") is bind.result
    assert bind.calls == [((name, {"M": 1}, (2, 1, 1), "sm_100a"), {})]


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
        cake_backend.clear_convenience_caches()
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


def test_gpu_default_path_is_bitwise_the_deterministic_path():
    """The former names with the default give the deterministic package's bits (K1 forward / input gradient /
    weight gradient, K2 forward / input gradient / weight gradient)."""
    device = _det_device_or_skip()
    g = torch.Generator(device=device).manual_seed(SEED)
    T, K, N = 1001, 6144, 576
    X, W, G = (
        _bf16((T, K), device, g),
        _bf16((N, K), device, g, 0.02),
        _bf16((T, N), device, g),
    )
    cases = [
        (cake_backend.projection_forward, det_backend.projection_det_forward, (X, W)),
        (cake_backend.projection_dgrad, det_backend.projection_det_dgrad, (G, W)),
        (cake_backend.projection_wgrad, det_backend.projection_det_wgrad, (G, X)),
    ]
    T2, K2, N2 = 1001, 6144, 256
    Xf = torch.randn(T2, K2, device=device, generator=g)
    Wf = torch.randn(N2, K2, device=device, generator=g) * 0.02
    Gf = torch.randn(T2, N2, device=device, generator=g)
    cases += [
        (cake_backend.router_forward, det_backend.router_det_forward, (Xf, Wf)),
        (cake_backend.router_dgrad, det_backend.router_det_dgrad, (Gf, Wf)),
        (cake_backend.router_wgrad, det_backend.router_det_wgrad, (Gf, Xf)),
    ]
    for ours, theirs, args in cases:
        try:
            expected = theirs(*args)
        except NotImplementedError as exc:
            pytest.skip(f"{theirs.__name__}: {exc}")
        got = ours(*args)
        assert got.dtype == expected.dtype and torch.equal(got, expected), ours.__name__


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


def _det_kernel(name: str) -> bool:
    """A K1 / K2 GEMM kernel of the deterministic package (its programs' symbols and its template bodies'
    namespace); the shared BF16x3 split stage is not one."""
    return "dense_projection_det_gemm" in name and not any(
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
    prepared = cake_backend.prepare_dense_projection_gemm(
        X.t(),
        G,
        out,
        transposed_out=True,
        deterministic=False,
        splits=splits,
        block_n=block_n,
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicGemm)
    assert (prepared.splits, prepared.plan.block_n) == (splits, block_n)
    return prepared


@pytest.fixture
def k1_rule(monkeypatch):
    """``k1_rule(G, X, out, splits)``: a verdict for the row (and no SM-count floor, excluded token counts, chain ranges
    or tables of exactly one SM count), so the convenience path runs this package's program at that split count;
    returns the template."""

    def force(G, X, out, splits, block_n=None):
        arch = det_backend.arch_for(out.device)
        N, K = int(G.shape[1]), int(X.shape[1])
        block_n = block_n or cake_backend.k1_block_n(N)
        f32 = out.dtype == torch.float32
        _require_template(arch, cake_backend.atomic_k1_template(block_n, f32))
        key = (arch, 1, K, N, "f32" if f32 else "bf16")
        monkeypatch.setitem(cake_backend.K1_MIN_SM_COUNT, arch, 0)
        monkeypatch.delitem(cake_backend.K1_RULE_MIN_SM_COUNT, key, raising=False)
        monkeypatch.delitem(
            cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS, key, raising=False
        )
        monkeypatch.delitem(
            cake_backend.K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS, key, raising=False
        )
        monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (block_n, splits))
        monkeypatch.setattr(cake_backend, "K1_SPLIT_OVERRIDES_BY_SM", {})
        monkeypatch.setattr(cake_backend, "K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM", {})
        monkeypatch.setattr(cake_backend, "K1_SPLIT_RANGE_OVERRIDES_BY_SM", {})
        monkeypatch.setattr(cake_backend, "K1_TWO_SPLIT_RANGES_BY_SM", {})
        monkeypatch.setattr(cake_backend, "K1_CHAIN_RANGES_BY_SM", {})
        cake_backend.clear_convenience_caches()
        return cake_backend.atomic_k1_template(block_n, f32)

    yield force
    cake_backend.clear_convenience_caches()


def _arena_tickets_zero():
    for arena in cake_backend._ATOMIC_ARENAS.values():
        buf = arena.buffers.get("k1_tickets")
        if buf is not None and buf.to(torch.int64).any():
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
    det_out = cake_backend.projection_wgrad(G, X, torch.empty_like(out))
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
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
        torch.cuda.synchronize()
        _k1_check(out, ref, f"convenience rerun {i}")
    assert _arena_tickets_zero()
    _assert_ran(
        lambda: cake_backend.projection_wgrad(G, X, out, deterministic=False),
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
    prepared = cake_backend.prepare_dense_projection_gemm(
        A,
        B,
        out,
        transposed_out=transposed_out,
        deterministic=False,
        splits=splits,
        block_n=192,
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicGemm)
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
            cake_backend.projection_wgrad(G, X, out, deterministic=False)
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
    prepared = cake_backend.prepare_dense_projection_gemm(
        X.t(),
        G,
        out,
        transposed_out=True,
        deterministic=False,
        block_n=block_n,
        chains=3,
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicGemm)
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
    cake_backend.clear_convenience_caches()
    try:
        out.fill_(float("nan"))
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
        torch.cuda.synchronize()
        assert torch.equal(out, first)
        _assert_ran(
            lambda: cake_backend.projection_wgrad(G, X, out, deterministic=False),
            prepared.module_name,
        )
    finally:
        cake_backend.clear_convenience_caches()
    short = torch.zeros(128, d_out, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="at one split"):
        cake_backend.prepare_dense_projection_gemm(
            torch.zeros(128, d_in, device=device, dtype=torch.bfloat16).t(),
            short,
            torch.empty(d_out, d_in, device=device),
            transposed_out=True,
            deterministic=False,
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
    prepared = cake_backend.prepare_dense_projection_gemm(
        A, B, out, deterministic=False, splits=2, block_n=256
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicGemm)
    assert (prepared.plan.L, prepared.splits, prepared.plan.swapped) == (H, 2, True)
    for _ in range(3):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, ref)
        assert prepared.counters_are_zero()
    # the planner keeps the batched rows on the deterministic program: the default convenience path matches it
    out2 = torch.empty_like(out)
    cake_backend.dense_projection_gemm(A, B, out2, deterministic=False)
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
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double(), f"T={T}")
        assert _arena_tickets_zero(), T
        if not cake_backend.k1_splits_fit(3, T):
            assert torch.equal(out, cake_backend.projection_wgrad(G, X))
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


def test_gpu_torch_deterministic_mode_rule(torch_determinism):
    """Under ``torch.use_deterministic_algorithms(True)`` an explicit ``deterministic=False`` raises before any
    launch; with ``warn_only=True`` it warns and returns the deterministic bits."""
    device = _det_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_hw", 4097, torch.bfloat16, device, SEED + 40)
    expected = cake_backend.projection_wgrad(G, X, torch.empty_like(out))
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running the deterministic path"):
        got = cake_backend.projection_wgrad(G, X, out, deterministic=False)
    assert torch.equal(got, expected)


def test_gpu_det_then_atomic_share_a_stream_not_a_workspace(k1_rule):
    """A deterministic call and a call of this package of one geometry on one stream: each package keeps its own
    arena (the deterministic package's slabs never reach this package's tickets)."""
    device, arch = _atomic_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 50)
    k1_rule(G, X, out, 2)
    ref = G.double().t() @ X.double()
    for _ in range(3):
        det = cake_backend.projection_wgrad(G, X, torch.empty_like(out))
        atomic = cake_backend.projection_wgrad(G, X, out, deterministic=False)
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
    """``deterministic=False`` K1 calls that run the deterministic program -- the forward, the input gradient (BF16
    and FP32 out), the weight gradient of rows without a verdict (both orientations, BF16 and FP32 out, another
    ``sk``), the batched head views of the generic entry -- go through this package's binding and give the
    deterministic package's bits: dense and padded outputs (the padding untouched), fresh tensors on every call, a
    larger row growing the arena (the smaller rows' bindings carved again), another stream with an arena of its own,
    the slice counters zero after the launches; a batched call the deterministic package runs on its swapped
    small-token program stays unbound (that package's call); a binding the deterministic package refuses raises its
    error."""
    device = _det_device_or_skip()
    cake_backend.clear_convenience_caches()
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
            # fmt: off
            calls = {
                "fwd": ((T, N), torch.bfloat16, lambda o, d: cake_backend.projection_forward(X, W, o, deterministic=d)),
                "dgrad": ((T, K), torch.bfloat16, lambda o, d: cake_backend.projection_dgrad(G, W, o, deterministic=d)),
                "dgrad_f32": ((T, K), torch.float32, lambda o, d: cake_backend.projection_dgrad(G, W, o, deterministic=d)),
                "wgrad": ((N, K), torch.bfloat16, lambda o, d: cake_backend.projection_wgrad(G, X, o, deterministic=d)),
                "wgrad_f32": ((N, K), torch.float32, lambda o, d: cake_backend.projection_wgrad(G, X, o, deterministic=d)),
                "wgrad_sk": ((N, K), torch.float32, lambda o, d: cake_backend.projection_wgrad(G, X, o, sk=False, deterministic=d)),
            }
            # fmt: on
            arch = det_backend.arch_for(device)
            sm_count = det_backend.device_facts(device).sm_count
            for op, ((rows_, cols), dtype, call) in calls.items():
                f32 = dtype == torch.float32
                if op.startswith("wgrad") and cake_backend.k1_plan(
                    arch=arch, L=1, M=K, N=N, K=T, out_f32=f32, sm_count=sm_count
                ):
                    continue  # a row with a verdict: this package's program
                if pad_out:
                    storage = torch.full(
                        (rows_, cols + 64), 7.0, device=device, dtype=dtype
                    )
                    out = storage[:, :cols]
                else:
                    out = torch.empty(rows_, cols, device=device, dtype=dtype)
                want = call(out, True).clone()
                for _ in range(2):
                    out.fill_(float("nan"))
                    assert call(out, False) is out
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
            want = cake_backend.dense_projection_gemm(A, B, out).clone()
            out.fill_(float("nan"))
            assert (
                cake_backend.dense_projection_gemm(A, B, out, deterministic=False)
                is out
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
        cake_backend.projection_forward(Xm, W)
    with pytest.raises(ValueError, match=re.escape(str(refused.value))):
        cake_backend.projection_forward(Xm, W, deterministic=False)


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
        assert cake_backend.projection_wgrad(G2, X2, out2, deterministic=False) is out2
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
        cake_backend.projection_forward(H, Wg, G, enable_pdl=True)
        cake_backend.projection_wgrad(G, X, out, deterministic=False, enable_pdl=True)
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double())
    assert _arena_tickets_zero()


def test_gpu_k1_bound_deterministic_program_waits_for_an_early_producer():
    """The deterministic program through this package's binding as a programmatic dependent launch: the
    deterministic forward before it (K = 8192) triggers its dependents at entry and writes ``G``; the input gradient
    with ``deterministic=False`` (bound, ``enable_pdl=True``) starts while it runs and must still read every row it
    wrote -- the deterministic package's bits -- and its slice counters end zero."""
    device = _det_device_or_skip()
    cake_backend.clear_convenience_caches()
    T, N, K = 4096, 128, 4096
    g = torch.Generator(device=device).manual_seed(SEED + 61)
    H = _bf16((T, 8192), device, g)
    Wg = _bf16((N, 8192), device, g, 8192**-0.5)
    W = _bf16((N, K), device, g, K**-0.5)
    G = torch.empty(T, N, device=device, dtype=torch.bfloat16)
    dX = torch.empty(T, K, device=device, dtype=torch.bfloat16)
    cake_backend.projection_forward(H, Wg, G, enable_pdl=False)
    want = cake_backend.projection_dgrad(G, W, enable_pdl=False).clone()
    for _ in range(3):
        G.fill_(float("nan"))
        dX.fill_(float("nan"))
        cake_backend.projection_forward(H, Wg, G, enable_pdl=True)
        assert (
            cake_backend.projection_dgrad(
                G, W, dX, deterministic=False, enable_pdl=True
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


@pytest.fixture
def router_rule(monkeypatch):
    """``router_rule(A, B, spec)``: a verdict for the router row of ``(A, B)``, so the default plan runs this
    package's program with the partition ``spec`` (or the deterministic program with ``None``)."""

    def force(op, A, B, spec):
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
        cake_backend.clear_convenience_caches()

    yield force
    cake_backend.clear_convenience_caches()


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
    prepared = cake_backend.prepare_router_fp32_gemm(
        A, B, out, splits=splits, deterministic=False
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
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
        det_backend.router_fp32_det_gemm(A[:, lo:hi], B[lo:hi, :], part, splits=1)
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
    """The head/tail partitions with fewer head units than CTA pairs (``"H<f>x<k>p<h>"``: the hidden 4096 verdicts of
    ``sm_103a`` and ``sm_100a``; 80 head units are more than a 152-SM part's CTA pairs, so there the partition is
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
        cake_backend.router_fp32_gemm(A, B, det_out, splits=1)
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
        (6144, 16172),
        (6144, 16231),
        (6144, 20480),
    ],
)
def test_gpu_router_tmem_head_tail_program_in_its_windows(monkeypatch, hidden, T):
    """On a part whose router table runs a head/tail window with the head/tail program with the A parts in tensor
    memory (212-SM sm_107a: hidden 5120 and 6144 at 16172 to 16384 tokens, ``ROUTER_HEAD_TAIL_TEMPLATES_BY_SM``;
    152-SM sm_103a, ``ROUTER_HEAD_TAIL_TEMPLATES``, with the windows that missed the accuracy rule forced,
    :data:`FAILED_ROUTER_WINDOWS`: hidden 4096 at 20480 to 49152 tokens, hidden 5120 at 16172 to 32768 and hidden
    6144 at 16172 to 20480) the default plan of the weight gradient (at the first and the last token
    count of each window and inside it) runs that program and meets the gate of :func:`test_gpu_router_d11_gate` over
    ``RERUNS`` launches into a NaN-poisoned ``out``: finite, within the reordering bound of the K-order sum of the
    deterministic program's one-split partials of its segments, relative Frobenius error against FP64 at most
    ``ROUTER_REL_FRO_MAX``.  On whole tiles (one unit per tile) it stores the bits of the three-chunk-buffer head/tail
    program at the same partition, which are the deterministic program's one-split result.  On a part of that
    architecture with another SM count, where the windows do not hold (``router_rules``: :data:`ROUTER_RULES` off
    :data:`ROUTER_SM_COUNT`, e.g. 148-SM sm_103a), the default plan is the deterministic program.  Skips on any other
    device and outside the part's windows."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    ts = cake_backend.router_head_tail_template(arch, sm_count)
    c3 = "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX
    if ts == c3:
        pytest.skip(f"{arch} with {sm_count} SMs runs {c3}")
    _force_failed_router_windows(
        monkeypatch, arch, lambda s: cake_backend.router_head_tail(s) is not None
    )
    spec = cake_backend._router_rule_spec(
        "nn_t", arch=arch, sm_count=sm_count, M=hidden, N=256, K=T
    )
    if cake_backend.router_head_tail(spec) is None:
        pytest.skip(
            f"no head/tail window of {arch} on {sm_count} SMs at {hidden}, T={T}"
        )
    X, W, G = _router_inputs(T, device, SEED + 30, K=hidden)
    A, B, out = _router_case("wgrad", X, W, G)
    if not cake_backend.router_rules(arch, sm_count)[1]:
        plan, reason = cake_backend.plan_atomic_router(
            A, B, out, arch=arch, sm_count=sm_count
        )
        assert plan is None, plan
        assert reason == "the planner keeps the row on the deterministic program"
        prepared = cake_backend.prepare_router_fp32_gemm(A, B, out, deterministic=False)
        assert not isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
        return
    _require_template(arch, ts)
    _require_template(arch, c3)
    plan, reason = cake_backend.plan_atomic_router(
        A, B, out, arch=arch, sm_count=sm_count
    )
    assert plan is not None and plan.template == ts and plan.head_spans > 0, reason
    prepared = cake_backend.prepare_router_fp32_gemm(A, B, out, deterministic=False)
    assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
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
    cake_backend.router_fp32_gemm(A, B, det_out, splits=1)
    assert torch.equal(stores[ts], stores[c3]) and torch.equal(stores[ts], det_out)


#: The SIMT programs' tiles ``(tile_n, tile_m, slices)`` a test forces (each runs where this checkout registers it).
SIMT_TILES = ((64, 64, 2), (128, 64, 2))


def _simt_spec(tile) -> str:
    return "simt_n{}m{}s{}".format(*tile)


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
    fixed order), and the prepared launch, the cached launch of ``router_fp32_gemm``, ``router_wgrad`` (its entry
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
    prepared = cake_backend.prepare_router_fp32_gemm(A, B, out, deterministic=False)
    assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
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
        cake_backend.router_fp32_gemm(A, B, cached, deterministic=False)
        entry = torch.full_like(out, float("nan"))
        cake_backend.router_wgrad(G, X, entry, deterministic=False)
        torch.cuda.synchronize()
        assert torch.equal(cached, first) and torch.equal(entry, first)
    _assert_ran(
        lambda: cake_backend.router_wgrad(G, X, entry, deterministic=False),
        prepared.module_name,
    )
    padded = torch.full((256, hidden + 4), float("nan"), device=device)
    cake_backend.router_wgrad(G, X, padded[:, :hidden], deterministic=False)
    torch.cuda.synchronize()
    assert torch.equal(padded[:, :hidden], first)
    assert bool(padded[:, hidden:].isnan().all())


@pytest.mark.parametrize("tile", SIMT_TILES, ids=_simt_spec)
def test_gpu_router_simt_program_replays_in_a_cuda_graph(router_rule, tile):
    """``router_wgrad`` in a SIMT window captured into a CUDA graph (its entry cache, the prepared launch) replays the
    eager bits into a NaN-poisoned ``out``; the activations and the gradient changed between replays are read
    again."""
    device, arch, template = _simt_device_or_skip(tile)
    hidden, T = 5120, 128
    X, W, G = _router_inputs(T, device, SEED + 41, K=hidden)
    A, B, out = _router_case("wgrad", X, W, G)
    router_rule("wgrad", A, B, _simt_spec(tile))
    eager = torch.empty_like(out)
    cake_backend.router_wgrad(G, X, eager, deterministic=False)
    prepared = cake_backend.prepare_router_fp32_gemm(G.t(), X, out, deterministic=False)
    assert prepared.plan.simt == tile
    entry_out = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cake_backend.router_wgrad(G, X, entry_out, deterministic=False)
            prepared.launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_backend.router_wgrad(G, X, entry_out, deterministic=False)
        prepared.launch()
    for scale in (1.0, -0.5):
        X.mul_(scale)
        want = torch.empty_like(out)
        cake_backend.router_wgrad(G, X, want, deterministic=False)
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
    want_y = det_backend.router_det_forward(X, W, enable_pdl=False)
    want_dx = det_backend.router_det_dgrad(want_y, W, enable_pdl=False)
    want_dw = torch.empty_like(dW)
    cake_backend.router_wgrad(
        want_y, want_dx, want_dw, deterministic=False, enable_pdl=False
    )
    torch.cuda.synchronize()
    assert (
        _rel_fro(want_dw, want_y.double().t() @ want_dx.double()) <= ROUTER_REL_FRO_MAX
    )
    for pdl in (True, False, None):
        for _ in range(3):
            for t in (Y, dX, dW):
                t.fill_(float("nan"))
            kw = dict(deterministic=False, enable_pdl=pdl)
            cake_backend.router_forward(X, W, Y, **kw)
            cake_backend.router_dgrad(Y, W, dX, **kw)
            cake_backend.router_wgrad(Y, dX, dW, **kw)
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
    launch, the cached launch of ``router_fp32_gemm``, ``router_dgrad`` (its entry cache), operands with padded rows
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
    prepared = cake_backend.prepare_router_fp32_gemm(A, B, out, deterministic=False)
    assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
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
        cake_backend.router_fp32_gemm(A, B, cached, deterministic=False)
        entry = torch.full_like(out, float("nan"))
        cake_backend.router_dgrad(G, W, entry, deterministic=False)
        torch.cuda.synchronize()
        assert torch.equal(cached, first) and torch.equal(entry, first)
    _assert_ran(
        lambda: cake_backend.router_dgrad(G, W, entry, deterministic=False),
        prepared.module_name,
    )
    G_pad = torch.full((T, 256 + 4), float("nan"), device=device)
    G_pad[:, :256] = G
    W_pad = torch.full((256, hidden + 8), float("nan"), device=device)
    W_pad[:, :hidden] = W
    padded = torch.full((T, hidden + 4), float("nan"), device=device)
    cake_backend.router_dgrad(
        G_pad[:, :256], W_pad[:, :hidden], padded[:, :hidden], deterministic=False
    )
    torch.cuda.synchronize()
    assert torch.equal(padded[:, :hidden], first)
    assert bool(padded[:, hidden:].isnan().all())


@pytest.mark.parametrize("tile", DGRAD_SIMT_TILES, ids=_simt_spec)
def test_gpu_router_dgrad_simt_program_replays_in_a_cuda_graph(router_rule, tile):
    """``router_dgrad`` in a SIMT window captured into a CUDA graph (its entry cache, the prepared launch) replays the
    eager bits into a NaN-poisoned ``out``; the output gradient changed between replays is read again."""
    device, arch, template = _dgrad_simt_device_or_skip(tile)
    hidden, T = 5120, 128
    X, W, G = _router_inputs(T, device, SEED + 141, K=hidden)
    A, B, out = _router_case("dgrad", X, W, G)
    router_rule("dgrad", A, B, _simt_spec(tile))
    eager = torch.empty_like(out)
    cake_backend.router_dgrad(G, W, eager, deterministic=False)
    prepared = cake_backend.prepare_router_fp32_gemm(G, W, out, deterministic=False)
    assert prepared.plan.simt == tile and prepared.plan.template == template
    entry_out = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            cake_backend.router_dgrad(G, W, entry_out, deterministic=False)
            prepared.launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cake_backend.router_dgrad(G, W, entry_out, deterministic=False)
        prepared.launch()
    for scale in (1.0, -0.5):
        G.mul_(scale)
        want = torch.empty_like(out)
        cake_backend.router_dgrad(G, W, want, deterministic=False)
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
    want_y = det_backend.router_det_forward(X, W, enable_pdl=False)
    want_dx = torch.empty_like(dX)
    cake_backend.router_dgrad(want_y, W, want_dx, deterministic=False, enable_pdl=False)
    torch.cuda.synchronize()
    assert _rel_fro(want_dx, want_y.double() @ W.double()) <= ROUTER_REL_FRO_MAX
    want_dw = torch.empty_like(dW)
    cake_backend.router_wgrad(
        want_y, want_dx, want_dw, deterministic=False, enable_pdl=False
    )
    torch.cuda.synchronize()
    for pdl in (True, False, None):
        for _ in range(3):
            for t in (Y, dX, dW):
                t.fill_(float("nan"))
            kw = dict(deterministic=False, enable_pdl=pdl)
            cake_backend.router_forward(X, W, Y, **kw)
            cake_backend.router_dgrad(Y, W, dX, **kw)
            cake_backend.router_wgrad(Y, dX, dW, **kw)
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
    """Unforced (``deterministic=False``, no split count): at 1 and 17 tokens and at the first and the last token
    count of every input-gradient SIMT window of this device's router table (``router_rules`` of its architecture and
    SM count), and one token past the last window, ``router_dgrad`` runs the program the planner gives -- inside a
    window the SIMT program of its tile (the prepared launch's program, within the gate, the same bits on every
    call), outside the deterministic program's bits."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rules, holds = cake_backend.router_rules(arch, sm_count)
    atomic = cake_backend.PreparedAtomicRouterGemm
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
            prepared = cake_backend.prepare_router_fp32_gemm(
                G, W, torch.empty(T, hidden, device=device), deterministic=False
            )
            det_out = cake_backend.router_dgrad(G, W)
            first = None
            for _ in range(2):
                out = torch.full((T, hidden), float("nan"), device=device)
                assert cake_backend.router_dgrad(G, W, out, deterministic=False) is out
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
                lambda G=G, W=W, out=out: cake_backend.router_dgrad(
                    G, W, out, deterministic=False
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


def test_gpu_router_entry_points_reach_the_stream_k_program(router_rule):
    """``router_forward`` / ``router_wgrad`` with ``deterministic=False`` and a verdict for each row (no forward
    verdict ships; the weight gradient of this hidden size runs the deterministic plan): the forward repeats bit
    for bit, and its top-8, like the deterministic forward's, is FP64's on every row but the near ties (the 8th and
    9th logits within 16 FP32 ulps); both meet the gate."""
    device, arch = _atomic_device_or_skip()
    X, W, G = _router_inputs(ROUTER_T, device, SEED + 21)
    router_rule("fwd", X, W.t(), "forward")
    router_rule("wgrad", G.t(), X, "pairs")
    first = cake_backend.router_forward(X, W, deterministic=False)
    det_fwd = cake_backend.router_forward(X, W)
    ref_fwd, ref_wgrad = X.double() @ W.double().t(), G.double().t() @ X.double()
    for name, out in (("atomic", first), ("det", det_fwd)):
        differ, ties = _top8_rows_differ(out, ref_fwd)
        print(
            f"ROUTER_TOP8 {arch} {name}: {differ} rows differ, {ties} near ties excluded"
        )
        assert differ == 0 and ties <= ROUTER_T // 100, (differ, ties)
    for _ in range(RERUNS):
        out = cake_backend.router_forward(X, W, deterministic=False)
        assert torch.equal(out, first) and _rel_fro(out, ref_fwd) <= ROUTER_REL_FRO_MAX
        wgrad = cake_backend.router_wgrad(G, X, deterministic=False)
        assert _rel_fro(wgrad, ref_wgrad) <= ROUTER_REL_FRO_MAX
    plan = cake_backend.plan_router_fp32_gemm(X, W.t(), first, deterministic=False)
    assert isinstance(plan, cake_backend.AtomicRouterPlan)
    _assert_ran(
        lambda: cake_backend.router_forward(X, W, first, deterministic=False),
        cake_jit.select_module(arch, plan.template),
    )


def test_gpu_planned_rows_run_the_programs_of_their_verdicts():
    """Unforced (``deterministic=False``, no split count): every K1 weight-gradient row with a verdict for this
    architecture and every measured router row prepares the program its verdict names on this device (by kernel
    name, at ``T = 16231``), and a row whose verdict is ``None`` -- or that this device's SM count leaves to the
    deterministic program -- prepares the deterministic launch."""
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
        prepared = cake_backend.prepare_projection_wgrad(G, X, out, deterministic=False)
        want = cake_backend.k1_plan(
            arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sm_count
        )
        if want is None:
            assert not isinstance(prepared, cake_backend.PreparedAtomicGemm), (M, N, dt)
            continue
        assert isinstance(prepared, cake_backend.PreparedAtomicGemm), (M, N, dt)
        assert (prepared.plan.block_n, prepared.plan.splits) == want
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
        prepared = cake_backend.prepare_router_fp32_gemm(A, B, out, deterministic=False)
        plan, _ = cake_backend.plan_atomic_router(
            A, B, out, arch=arch, sm_count=sm_count
        )
        if plan is None:
            assert not isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
            continue
        assert spec is not None
        assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
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
    """Unforced (``deterministic=False``, no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_OVERRIDES_BY_SM` (``sm_107a``: 216 SMs): every overridden row prepares and runs the override's
    program at the first and the last token count of :data:`K1_SPLIT_RULES_BY_SM_TOKENS` outside the part's ranges of
    :data:`K1_CHAIN_RANGES_BY_SM`, which come first (``4096 -> 1024`` FP32 at 16384 tokens: the 192-wide row-major-store
    program at one split), meets the deterministic package's FP64 budget and, where both run one split, gives the bits
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
        for T in cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS:
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
            prepared = cake_backend.prepare_projection_wgrad(
                G, X, out, deterministic=False
            )
            assert isinstance(prepared, cake_backend.PreparedAtomicGemm), (key, T)
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


def test_gpu_k1_sm_count_overrides_through_the_convenience_binding():
    """Unforced (``deterministic=False``, no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_OVERRIDES_BY_SM` (``sm_107a``: 216 SMs): the convenience path of every overridden row binds the
    override's program (its :data:`_K1_ATOMIC_CACHE` entry) at the first and the last token count of
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS` outside the part's ranges of :data:`K1_CHAIN_RANGES_BY_SM` and gives the bits of
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
        for T in cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS:
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
            prepared = cake_backend.prepare_projection_wgrad(
                G, X, want, deterministic=False
            )
            prepared.launch()
            cake_backend.clear_convenience_caches()
            out = torch.full_like(want, float("nan"))
            assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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
    and one token outside it (``want``: the row's :data:`K1_SPLIT_RULES` verdict); ``choice`` is the planner's."""
    rows = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM.get((arch, sm_count), {})
    cases = []
    for key, spans in sorted(rows.items()):
        _, L, M, N, dt = key
        rule = cake_backend.K1_SPLIT_RULES[key]
        held = lambda T, spans=spans: any(  # noqa: E731
            lo <= T and (hi is None or hi >= T) for lo, hi in spans
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


def _k1_with_chain_ranges(arch, sm_count, key, T, want):
    """``(want, chains)``: an edge case's expected ``(block_n, splits)`` and chain count with
    :data:`K1_CHAIN_RANGES_BY_SM`, whose ranges come before the tables the case is drawn from
    (:func:`k1_chain_plan`): a token count of a chain range expects that range's tile width at one split and its chain
    count, every other the case's ``want`` (one chain)."""
    _, L, M, N, dt = key
    chain = cake_backend.k1_chain_plan(
        arch=arch, L=L, M=M, N=N, K=T, out_f32=dt == "f32", sm_count=sm_count
    )
    if chain is None:
        return want, 1
    return (chain[0], 1), chain[1]


def test_gpu_k1_sm_count_token_ranges_run_at_their_edges():
    """Unforced (``deterministic=False``, no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` or
    :data:`K1_TWO_SPLIT_RANGES_BY_SM` (``sm_107a``: 212 SMs; ``sm_100a`` / ``sm_103a``: 152 SMs): at the first and the
    last token count of every range of a row, and one token past the first token count of a two-split range (an odd
    number of K blocks), the prepared launch runs the program of the part's table in :data:`K1_SPLIT_RULES_BY_SM` (of
    the range, for an override or two-split range) and meets the K1 accuracy rule (:func:`_k1_check`); one token outside
    a range (and outside the window) the row runs the deterministic program, with the deterministic package's bits.  On
    a part of exactly the SM count of a table of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` (148-SM ``sm_100a`` /
    ``sm_103a`` parts) the reverse: the deterministic program with its bits at both ends of a range, the row's program
    one token outside it.  An edge inside a range of :data:`K1_CHAIN_RANGES_BY_SM` for the part runs that range's
    program of several chains instead (:func:`_k1_with_chain_ranges`), under the same FP64 rule.  A part of any other SM
    count skips."""
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
        prepared = cake_backend.prepare_projection_wgrad(G, X, out, deterministic=False)
        if want_plan is not None:
            assert plan == want_plan, (key, T)
            assert isinstance(prepared, cake_backend.PreparedAtomicGemm), (key, T)
            assert (prepared.plan.block_n, prepared.plan.splits) == plan, (key, T)
            assert prepared.plan.chains == chains, (key, T)
            assert prepared.template == cake_backend.atomic_k1_template(
                plan[0], f32, chains
            ), (key, T)
            _assert_ran(prepared.launch, prepared.module_name)
            torch.cuda.synchronize()
            _k1_check(out, G.double().t() @ X.double(), f"{M} {N} {dt} {T}")
            ran += 1
        else:
            assert plan is None, (key, T)
            assert not isinstance(prepared, cake_backend.PreparedAtomicGemm), (key, T)
            prepared.launch()
            want = torch.full_like(out, float("nan"))
            det_backend.projection_det_wgrad(G, X, want)
            torch.cuda.synchronize()
            assert torch.equal(out, want), (key, T)
    print(
        f"K1_TOKEN_RANGES {arch} sm_count={sm_count}: {ran} launches ran this package's programs at range edges"
    )
    assert ran


def test_gpu_k1_sm_count_token_ranges_through_the_convenience_binding():
    """Unforced (``deterministic=False``, no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` or
    :data:`K1_TWO_SPLIT_RANGES_BY_SM` (``sm_107a``: 212 SMs; ``sm_100a`` / ``sm_103a``: 152 SMs): the
    convenience path binds the part's program
    (the range's, for an override or two-split range) at the first and the last token count of every range of a row
    (its :data:`_K1_ATOMIC_CACHE` entry) and gives the bits of the prepared launch where at most two partials meet (the
    FP64 budget otherwise), the tickets zero after; one token outside a range, and one token past the first token count
    of a two-split range, it binds the deterministic package's plan.  On a part of exactly the SM count of a table
    of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` the reverse.  An edge inside a range of :data:`K1_CHAIN_RANGES_BY_SM`
    for the part binds that range's program of several chains instead (one split: the bits of the prepared launch).  A
    part of any other SM count skips."""
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
        prepared = cake_backend.prepare_projection_wgrad(
            G, X, want, deterministic=False
        )
        prepared.launch()
        cake_backend.clear_convenience_caches()
        out = torch.full_like(want, float("nan"))
        assert cake_backend.projection_wgrad(G, X, out, deterministic=False) is out
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


def test_gpu_router_token_windows_run_at_their_edges():
    """Unforced (``deterministic=False``, no split count): at the first and the last token count of every token window
    of a router weight-gradient row of this architecture (the smallest measured count, 64, for a window that starts
    at one token) the prepared launch runs the program the planner gives -- the SIMT program of a SIMT window and
    the partition of a head/tail window on the measured SM count, a "waves" window's partition where
    :func:`cake_backend.router_waves` gives one and no SIMT window holds the token count -- and meets the gate; one
    token outside every window, or on another SM count, it runs the deterministic launch."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    rules, measured = cake_backend.router_rules(arch, sm_count)
    atomic = cake_backend.PreparedAtomicRouterGemm
    ran = 0
    for (rule_arch, layout, hidden, _experts), rule in sorted(
        rules.items(), key=lambda kv: kv[0]
    ):
        if rule_arch != arch or layout != "nn_t" or not isinstance(rule, tuple):
            continue
        for first, last, spec in rule:
            for T, inside in (
                (first - 1, False),
                (max(first, 64), True),
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
                prepared = cake_backend.prepare_router_fp32_gemm(
                    A, B, out, deterministic=False
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
    """``deterministic=False`` without a split count on a router row without a verdict (the forward; the weight
    gradient at a token count outside every window): the deterministic program's bits (its production split counts),
    through this package's binding of it (``_K2_SPLITK_CACHE``); this package plans nothing for the call."""
    device, arch = _atomic_device_or_skip()
    for hidden in (4096, 5120, 6144):
        X, W, G = _router_inputs(8000, device, SEED + 23, K=hidden)
        cake_backend.clear_convenience_caches()
        fwd = cake_backend.router_forward(X, W, deterministic=False)
        wgrad = cake_backend.router_wgrad(G, X, deterministic=False)
        assert not cake_backend._K2_ATOMIC_CACHE
        assert len(cake_backend._K2_SPLITK_CACHE) == 2
        assert torch.equal(fwd, cake_backend.router_forward(X, W))
        assert torch.equal(wgrad, cake_backend.router_wgrad(G, X))
        A, B, out = _router_case("fwd", X, W, G)
        assert cake_backend.router_fp32_gemm(A, B, out, deterministic=False) is out
        assert torch.equal(out, fwd) and not cake_backend._K2_ATOMIC_CACHE


def test_gpu_router_windows_failing_the_accuracy_rule_give_the_deterministic_bits():
    """Unforced (``deterministic=False``, no split count): at the first and the last token count of each router window
    of this architecture that missed the accuracy rule (:func:`_failed_router_window_rows`; 64 for a window that starts
    at one token) where this device's planner gives the row neither a partition nor a SIMT program -- every one of
    them on 148-SM and 152-SM parts except the hidden 5120 "waves" windows -- the weight gradient and the input
    gradient give the deterministic program's bits through this package's binding, and no launch of this package's
    programs is prepared (a row whose shipped window holds the token count keeps a cached entry without a plan)."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    ran = 0
    for (rule_arch, _sms, layout, hidden), windows in sorted(
        _failed_router_window_rows()
    ):
        if rule_arch != arch:
            continue
        for first, last, _spec in windows:
            for T in sorted({max(first, 64), last}):
                M, N, K = _router_launch_dims(layout, hidden, 256, T)
                kw = dict(arch=arch, sm_count=sm_count, M=M, N=N, K=K)
                if (
                    cake_backend.router_partition(layout, **kw) is not None
                    or cake_backend.router_simt(layout, **kw) is not None
                ):
                    assert sm_count not in (148, 152) or (
                        (layout, hidden) == ("nn_t", 5120) and T <= 160
                    ), (layout, hidden, T)
                    continue
                X, W, G = _router_inputs(T, device, SEED + 150, K=hidden)
                cake_backend.clear_convenience_caches()
                if layout == "nn_t":
                    got = cake_backend.router_wgrad(G, X, deterministic=False)
                    want = cake_backend.router_wgrad(G, X)
                else:
                    got = cake_backend.router_dgrad(G, W, deterministic=False)
                    want = cake_backend.router_dgrad(G, W)
                torch.cuda.synchronize()
                cached = cake_backend._K2_ATOMIC_CACHE.values()
                assert all(entry.plan is None for entry in cached), (layout, hidden, T)
                assert torch.equal(got, want), (layout, hidden, T)
                ran += 1
    print(f"ROUTER_FAILED_WINDOWS {arch} sm_count={sm_count}: {ran} token counts ran")
    if not ran:
        pytest.skip(f"no router window of {arch} missed the accuracy rule")


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
    # no verdict at 1000 tokens on any architecture and SM count (the hidden-6144 weight-gradient windows, sm_103a on
    # 152 SMs and sm_107a on 212 SMs, start at 16172 tokens)
    hidden, T = 6144, 1000
    X, W, G = _router_inputs(T, device, SEED + 26, K=hidden)
    keys = {"fwd": (arch, "kk", hidden, 256), "wgrad": (arch, "nn_t", hidden, 256)}
    for layout, M, K in (("kk", T, hidden), ("nn_t", hidden, T)):
        spec = cake_backend._router_rule_spec(
            layout, arch=arch, sm_count=sm_count, M=M, N=256, K=K
        )
        assert spec is None, (layout, spec)
    cake_backend.clear_convenience_caches()
    want_fwd = det_backend.router_fp32_det_gemm(
        X, W.t(), torch.empty(T, 256, device=device)
    )
    want_wgrad = det_backend.router_fp32_det_gemm(
        G.t(), X, torch.empty(256, hidden, device=device)
    )

    def unplanned():
        for _ in range(2):
            fwd = cake_backend.router_forward(X, W, deterministic=False)
            wgrad = cake_backend.router_wgrad(G, X, deterministic=False)
            assert torch.equal(fwd, want_fwd) and torch.equal(wgrad, want_wgrad)
        assert not cake_backend._K2_ATOMIC_CACHE

    unplanned()
    monkeypatch.setitem(table, keys["fwd"], "forward")
    monkeypatch.setitem(table, keys["wgrad"], "pairs")
    ref_fwd, ref_wgrad = X.double() @ W.double().t(), G.double().t() @ X.double()

    def fwd():
        return cake_backend.router_forward(X, W, deterministic=False)

    def wgrad():
        return cake_backend.router_wgrad(G, X, deterministic=False)

    for op, A, B, ref, call in (
        ("fwd", X, W.t(), ref_fwd, fwd),
        ("wgrad", G.t(), X, ref_wgrad, wgrad),
    ):
        out = torch.empty(A.shape[0], B.shape[1], device=device)
        plan = cake_backend.plan_router_fp32_gemm(A, B, out, deterministic=False)
        assert isinstance(plan, cake_backend.AtomicRouterPlan), op
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
    -- through ``router_wgrad`` and through ``router_fp32_gemm`` -- runs the program the planner gives under the new
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
        lambda: cake_backend.router_wgrad(G, X, out, deterministic=False),
        lambda: cake_backend.router_fp32_gemm(A, B, out, deterministic=False),
    )
    cake_backend.clear_convenience_caches()
    plans = []
    for spec in ("pairs", "S2" if simt is None else _simt_spec(simt)):
        monkeypatch.setitem(table, (arch, "nn_t", hidden, 256), ((1, T, spec),))
        plan = cake_backend.plan_router_fp32_gemm(A, B, out, deterministic=False)
        assert isinstance(plan, cake_backend.AtomicRouterPlan), spec
        for call in calls:
            _assert_ran(call, cake_jit.select_module(arch, plan.template))
            assert _rel_fro(call(), ref) <= ROUTER_REL_FRO_MAX, spec
        cached = [e.plan for e in cake_backend._K2_ATOMIC_CACHE.values()]
        assert cached and all(p == plan for p in cached), spec
        plans.append(plan)
    assert plans[0] != plans[1]


def test_gpu_router_deterministic_program_through_the_binding():
    """``deterministic=False`` calls that run the deterministic program -- the forward and the input gradient, the
    weight gradient of a row without a verdict, the generic ``kn`` call -- go through this package's binding
    (``_K2_SPLITK_CACHE``; nothing in ``_K2_ATOMIC_CACHE``) and give the deterministic package's bits: dense and
    padded outputs (one split stores into a padded ``out`` through its flat alias, several splits reduce into it), a
    split source with padded rows (the torch chain), repeated calls; a larger row grows the arena and the smaller
    rows' launches are carved again with the same bits; another stream binds an arena of its own; a binding the
    deterministic package refuses raises its error."""
    device, arch = _atomic_device_or_skip()
    # no router verdict on any architecture from 257 to 16171 tokens (the SIMT windows end at 256 tokens or before)
    hidden = 6144
    cake_backend.clear_convenience_caches()

    def check(T, seed, pad_out=False, pad_g=False):
        X, W, G = _router_inputs(T, device, seed, K=hidden)
        if pad_g:
            G = torch.cat([G, torch.zeros(T, 16, device=device)], 1)[:, :256]
        shapes = {"fwd": (T, 256), "dgrad": (T, hidden), "wgrad": (256, hidden)}
        # fmt: off
        calls = {
            "fwd": lambda out, d: cake_backend.router_forward(X, W, out, deterministic=d),
            "dgrad": lambda out, d: cake_backend.router_dgrad(G, W, out, deterministic=d),
            "wgrad": lambda out, d: cake_backend.router_wgrad(G, X, out, deterministic=d),
        }
        # fmt: on
        for op, call in calls.items():
            rows, cols = shapes[op]
            if pad_out:
                storage = torch.full((rows, cols + 64), 7.0, device=device)
                out = storage[:, :cols]
            else:
                out = torch.empty(rows, cols, device=device)
            want = call(out, True).clone()
            for _ in range(2):
                out.fill_(float("nan"))
                assert call(out, False) is out
                assert torch.equal(out, want), (op, T, pad_out, pad_g)
            if pad_out:
                assert bool((storage[:, cols:] == 7.0).all()), op
        out = torch.empty(T, hidden, device=device)
        want = det_backend.router_fp32_det_gemm(G, W, out.clone())
        assert cake_backend.router_fp32_gemm(G, W, out, deterministic=False) is out
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
        cake_backend.router_dgrad(G, W, out)
    with pytest.raises(ValueError, match=re.escape(str(refused.value))):
        cake_backend.router_dgrad(G, W, out, deterministic=False)


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
    cake_backend.clear_convenience_caches()
    want_fwd = det_backend.router_det_forward(X, W, enable_pdl=False)
    want_wgrad = det_backend.router_det_wgrad(G, X, enable_pdl=False)
    Y = torch.empty(T, 256, device=device)
    dW = torch.empty(256, hidden, device=device)
    stream = torch.cuda.Stream(device)

    def step():
        cake_backend.router_forward(X, W, Y, deterministic=False)
        cake_backend.router_wgrad(G, X, dW, deterministic=False)

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
    cake_backend.clear_convenience_caches()
    want_y = det_backend.router_det_forward(X, W, enable_pdl=False)
    want_dx = det_backend.router_det_dgrad(want_y, W, enable_pdl=False)
    want_dw = det_backend.router_det_wgrad(want_y, want_dx, enable_pdl=False)
    Y = torch.empty(T, 256, device=device)
    dX = torch.empty(T, hidden, device=device)
    dW = torch.empty(256, hidden, device=device)
    for pdl in (True, False, None):
        for _ in range(3):
            for t in (Y, dX, dW):
                t.fill_(float("nan"))
            kw = dict(deterministic=False, enable_pdl=pdl)
            cake_backend.router_forward(X, W, Y, **kw)
            cake_backend.router_dgrad(Y, W, dX, **kw)
            cake_backend.router_wgrad(Y, dX, dW, **kw)
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
            cake_backend.router_fp32_gemm(A, B, out, splits=4, deterministic=False)
        for (A, B, out), splits in cases:
            plan = cake_backend.plan_router_fp32_gemm(
                A, B, out, splits=splits, deterministic=False
            )
            if not isinstance(plan, cake_backend.AtomicRouterPlan):
                continue
            prepared = cake_backend.prepare_router_fp32_gemm(
                A, B, out.clone(), splits=splits, deterministic=False
            )
            want = prepared.launch().clone()
            got = cake_backend.router_fp32_gemm(
                A, B, out, splits=splits, deterministic=False
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
    """Unforced (``deterministic=False``, no split count): at the first and the last token count of every head/tail
    window of this device's router table (``router_rules`` of its architecture and SM count, with the windows that
    missed the accuracy rule forced, :data:`FAILED_ROUTER_WINDOWS`), the convenience path's cached launch (its bound arguments, ``_K2Binding``) runs
    the prepared launch's plan and program, meets the gate and, for a reproducible plan, gives the prepared launch's
    bits; the second call of the key reuses the binding."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    _force_failed_router_windows(
        monkeypatch, arch, lambda s: cake_backend.router_head_tail(s) is not None
    )
    rules, holds = cake_backend.router_rules(arch, sm_count)
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
        prepared = cake_backend.prepare_router_fp32_gemm(
            A, B, out.clone(), deterministic=False
        )
        assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm), (hidden, T)
        assert prepared.plan.head_spans > 0, (hidden, T)
        want = prepared.launch().clone()
        cake_backend.clear_convenience_caches()
        _assert_ran(
            lambda A=A, B=B, out=out: cake_backend.router_fp32_gemm(
                A, B, out, deterministic=False
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
        assert cake_backend.router_fp32_gemm(A, B, out, deterministic=False) is out
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
        lambda: cake_backend.router_fp32_gemm(
            A, B, out, splits=splits, deterministic=False
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
        det_backend.plan_router_fp32_det_gemm(X, W.t(), out, splits=1)
    message = re.escape(str(refused.value))
    with pytest.raises(ValueError, match=message):
        cake_backend.plan_router_fp32_gemm(X, W.t(), out, splits=1, deterministic=False)
    with pytest.raises(ValueError, match=message):
        cake_backend.prepare_router_fp32_gemm(
            X, W.t(), out, splits=1, deterministic=False
        )
    with pytest.raises(ValueError, match=message):
        cake_backend.router_fp32_gemm(X, W.t(), out, splits=1, deterministic=False)


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
    """``deterministic=False`` through ``_K2_ENTRY_CACHE``: every router entry point's call -- the forward with and
    without a split count, the input gradient, the weight gradient with and without one; dense and padded outputs;
    another stream -- binds the launch its own path takes and gives that path's results on repeated calls: the
    deterministic program and a reproducible plan bit for bit (the deterministic program the deterministic package's
    bits), every launch within the gate; the padding keeps its values."""
    device, arch = _atomic_device_or_skip()
    cake_backend.clear_convenience_caches()
    X, W, G = _router_inputs(T, device, SEED + 100 + T, K=hidden)
    tensors = {"fwd": (X, W), "dgrad": (G, W), "wgrad": (G, X)}
    calls = {
        "fwd": cake_backend.router_forward,
        "dgrad": cake_backend.router_dgrad,
        "wgrad": cake_backend.router_wgrad,
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
        kw = dict(deterministic=False)
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
                det_out = calls[op](P, Q, torch.empty_like(out), **det_kw)
                stream.synchronize()
                assert torch.equal(out, det_out), (op, splits, pad)
        ran["bits" if _router_entry_bits(entry) else "gate"] += 1
    print(f"ROUTER_ENTRY {arch} H{hidden} T{T}: {ran}")


def test_gpu_router_entry_points_capture_in_a_cuda_graph(monkeypatch):
    """The router entry points' ``deterministic=False`` calls through ``_K2_ENTRY_CACHE`` (the forward with and
    without a split count, the input gradient, the weight gradient of a row with a verdict and of one without), their
    keys cached on the capturing stream, capture in a CUDA graph without planning: every replay gives the eager calls'
    results -- bit for bit for the deterministic program and a reproducible plan -- within the gate."""
    device, arch = _atomic_device_or_skip()
    cake_backend.clear_convenience_caches()
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
                cake_backend.router_forward(
                    P, Q, out, splits=splits, deterministic=False
                )
            elif op == "dgrad":
                cake_backend.router_dgrad(P, Q, out, deterministic=False)
            else:
                cake_backend.router_wgrad(P, Q, out, splits=splits, deterministic=False)

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
    cake_backend.clear_convenience_caches()
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
                cake_backend.projection_wgrad(G, X, out, deterministic=False)
                cake_backend.router_wgrad(Gf, Xf, rout, deterministic=False)
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
