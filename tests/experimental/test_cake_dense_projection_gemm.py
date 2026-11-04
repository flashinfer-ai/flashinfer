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
from types import ModuleType

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
    """``deterministic=False``: the router weight gradient reaches this package's stream-K path, and so does the
    forward with a split count or while some router table gives the forward a verdict; without either the forward
    runs the deterministic program through this package's binding (the plan the planner keeps for a row without a
    verdict).  The input gradient runs the deterministic program (one split) through that binding too, the split stage
    the deterministic package's.  Each entry point passes its own tensors as the launch views."""
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
    assert cake_backend.router_dgrad(G, W, deterministic=False) is splitk.result
    assert not dgrad.calls
    dg_args, dg_kwargs = splitk.calls[1]
    assert dg_args[0] is G and dg_args[1] is W and dg_args[3] == 1 and not dg_kwargs
    assert dg_args[5] is None
    assert tuple(dg_args[2].shape) == (T, K) and dg_args[2].dtype == torch.float32
    _same_entry_views(dg_args[4], G.unsqueeze(0), W, cake_backend._DGRAD_VIEWS)
    assert (
        cake_backend.split_fp32_to_bf16x3_("s", "p", "r", deterministic=False)
        is split.result
    )
    calls = atomic.calls
    (fwd2_args, fwd2_kwargs), (fwd_args, fwd_kwargs), (wg_args, wg_kwargs) = calls
    assert fwd2_args[1] is X and fwd2_args[2].equal(W.t())
    assert fwd2_kwargs["splits"] == 2
    assert (
        fwd_args[0] == "launch"
        and fwd_args[1] is X
        and fwd_args[2].equal(W.t())
        and fwd_kwargs["splits"] is None
    )
    for kw in (fwd2_kwargs, fwd_kwargs, wg_kwargs):
        assert set(kw) == {"splits", "enable_pdl", "views"} and kw["enable_pdl"] is None
    for kw in (fwd2_kwargs, fwd_kwargs):
        _same_entry_views(kw["views"], X.unsqueeze(0), W, cake_backend._FORWARD_VIEWS)
    assert tuple(fwd_args[3].shape) == (T, N) and fwd_args[3].dtype == torch.float32
    assert wg_args[1].equal(G.t()) and wg_args[2] is X and wg_kwargs["splits"] == 5
    _same_entry_views(wg_kwargs["views"], X.unsqueeze(0), G, cake_backend._WGRAD_VIEWS)
    assert tuple(wg_args[3].shape) == (N, K)
    # the generic entry: the kn class runs the deterministic program (this package's binding), kk / nn_t this
    # package's path, without entry-point views
    assert (
        cake_backend.router_fp32_gemm(
            G, W, torch.zeros(T, K), deterministic=False, splits=1
        )
        is splitk.result
    )
    assert splitk.calls[2][0][3] == 1 and splitk.calls[2][0][4] is None
    assert splitk.calls[2][0][5] is None
    assert (
        cake_backend.router_fp32_gemm(X, W.t(), torch.zeros(T, N), deterministic=False)
        is atomic.result
    )
    assert (
        cake_backend.router_fp32_gemm(G.t(), X, torch.zeros(N, K), deterministic=False)
        is atomic.result
    )
    assert all(kw["views"] is None for _, kw in atomic.calls[3:])
    assert not gemm.calls and len(splitk.calls) == 3 and len(atomic.calls) == 5


@pytest.mark.parametrize("pdl", [True, False])
def test_k2_routes_pass_enable_pdl(monkeypatch, pdl):
    """``enable_pdl`` reaches every K2 route of ``deterministic=False``: this package's binding of the deterministic
    program (the forward without a verdict, the input gradient, the generic ``kn`` call) and this package's stream-K
    path (the forward with a split count, the weight gradient)."""
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
    assert [args[5] for args, _ in splitk.calls] == [pdl] * 3
    assert [kwargs["enable_pdl"] for _, kwargs in atomic.calls] == [pdl] * 2


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
    """The verdicts of the three architectures take 28 / 32 / 26 of the 124 weight-gradient rows (the MLA head rows,
    ``L = 64``, run the deterministic program), at 16172 and 16231 tokens; on 148 SMs sm_103a leaves ``4096 -> 48`` to
    the deterministic program (28 rows).  On sm_107a the token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` run the
    deterministic program (seven FP32 rows and one BF16 row at 16231 tokens on 216 SMs).  The per-SM-count sm_107a
    tables take 19 rows on 212 SMs and 14 on 208 SMs; an SM count below 216 without a table takes none."""
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
    # d_out = 128 at d_in 4096 was measured and missed on every architecture
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
    assert cake_backend.k1_plan(
        arch="sm_107a", M=4096, N=512, sm_count=216, L=1, K=16172, out_f32=True
    ) == (128, 1)
    # ... and the deterministic program at the token counts of K1_SPLIT_RULE_EXCLUDED_TOKENS
    assert (
        cake_backend.k1_plan(
            arch="sm_107a", M=4096, N=512, sm_count=216, L=1, K=16231, out_f32=True
        )
        is None
    )
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


def test_k1_excluded_token_counts():
    """A row of :data:`K1_SPLIT_RULES` or of a table of :data:`K1_SPLIT_RULES_BY_SM` runs the deterministic program at
    the token counts of its :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` entries: from the entry's SM count up where that
    count is at or above the row's SM-count floor, on exactly that SM count below it.  Every entry belongs to a row
    with a verdict on its part; per SM count its ranges are closed, disjoint and not adjacent, start at a measured
    token count (512 to 65536; the sm_107a FP32 budget ranges one token above one) and end at one or have no upper
    bound; the row runs the deterministic program at both ends and keeps its verdict (inside the token window, the
    override of :data:`K1_SPLIT_OVERRIDES_BY_SM` for the part's SM count, if any) one token outside a range (inside the
    token window of a per-SM-count table), on a part with fewer SMs than an entry at or above the floor, on the floor's
    part for an entry below it, and above 65536 tokens unless a range has no upper bound."""
    table = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
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
            for sms, a, b in table[key]:
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
            assert 512 <= a <= 65536, key
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
                    assert got == kept(sms - 1, T, want), (key, T)
            if sms < floor and cake_backend.K1_SPLIT_RULES.get(key) is not None:
                for T in ends:
                    if not cut(floor, T):
                        got = cake_backend.k1_plan(K=T, sm_count=floor, **kw)
                        rule = tuple(cake_backend.K1_SPLIT_RULES[key])
                        assert got == kept(floor, T, rule), (key, T)
        top = max(floor, *(e[0] for e in entries))
        verdict = cake_backend.K1_SPLIT_RULES.get(key)
        if verdict is not None and not any(e[2] is None for e in entries):
            got = cake_backend.k1_plan(K=65537, sm_count=top, **kw)
            assert got == tuple(verdict), key
    kw = dict(arch="sm_100a", L=1, M=4096, N=32, out_f32=False)
    assert cake_backend.k1_plan(K=512, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=511, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=512, sm_count=151, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=131072, sm_count=152, **kw) == (64, 2)
    kw = dict(arch="sm_103a", L=1, M=4096, N=32, out_f32=False)
    assert cake_backend.k1_plan(K=512, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=511, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=512, sm_count=151, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=131072, sm_count=152, **kw) == (64, 2)
    kw = dict(arch="sm_103a", L=1, M=5120, N=32, out_f32=False)
    assert cake_backend.k1_plan(K=512, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=511, sm_count=152, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=512, sm_count=151, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=32768, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=131072, sm_count=152, **kw) is None
    assert cake_backend.k1_plan(K=131072, sm_count=149, **kw) == (64, 2)
    # a row without an entry keeps its verdict at every token count its split count fits
    kw = dict(arch="sm_100a", L=1, M=4096, N=32, out_f32=True, sm_count=152)
    assert ("sm_100a", 1, 4096, 32, "f32") not in table
    for T in (65, 512, 16231, 65536, 131072):
        assert cake_backend.k1_plan(K=T, **kw) == (64, 2), T


def test_k1_rule_table_entries_decide_the_row(monkeypatch):
    # 16172 tokens: 16231 is a 216-SM token count of K1_SPLIT_RULE_EXCLUDED_TOKENS for sm_107a 4096 -> 32 FP32
    kw = dict(L=1, M=4096, N=32, K=16172, out_f32=True)
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
def test_k1_per_sm_count_tables_hold_only_at_their_token_counts(sm_count):
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


def test_k1_212_sm_token_ranges():
    """On 212-SM sm_107a parts the rows of :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` take their 212-SM verdict at
    every token count of their ranges (both ends and the measured token counts inside), except a token count of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` for the part, and run the deterministic program one token outside a range
    and at every measured token count outside the ranges and the window.  Every range starts and ends at a measured
    token count, lies inside 512 to 65536 tokens and belongs to a row with a program on that part.  Another SM count
    (208, 210, 216, 217) and every other row keep their routing: 208 SMs and the other 212-SM rows run the
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
        for T in K1_212_MEASURED_TOKENS:
            if T not in held and not first <= T <= last:
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


#: The 148-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`, ``(arch, d_in, d_out, out_dtype) -> ranges``.
K1_148_EXCLUDED_TOKENS = {
    ("sm_100a", 4096, 48, "bf16"): ((512, 512),),
    ("sm_100a", 4096, 48, "f32"): ((512, 512),),
    ("sm_100a", 4096, 64, "bf16"): ((512, 512), (65536, 65537)),
    ("sm_100a", 4096, 64, "f32"): ((512, 512), (65536, 65537)),
    ("sm_100a", 5120, 48, "bf16"): ((512, 2048),),
    ("sm_100a", 5120, 48, "f32"): ((512, 512),),
    ("sm_100a", 5120, 64, "bf16"): ((512, 2048), (8192, 8192)),
    ("sm_100a", 5120, 64, "f32"): ((512, 1024), (8192, 8192)),
    ("sm_103a", 4096, 32, "bf16"): ((65536, 65537),),
    ("sm_103a", 4096, 32, "f32"): ((65536, 65537),),
    ("sm_103a", 5120, 32, "bf16"): ((32767, None),),
    ("sm_103a", 5120, 48, "bf16"): ((512, 2048), (32767, 32768)),
    ("sm_103a", 5120, 48, "f32"): ((512, 2048),),
    ("sm_103a", 5120, 64, "bf16"): ((512, 1024), (32767, 32769)),
    ("sm_103a", 5120, 64, "f32"): ((512, 1024),),
    ("sm_103a", 5120, 128, "bf16"): ((512, 4096), (32767, None)),
    ("sm_103a", 5120, 128, "f32"): ((512, 4096), (32768, 65536)),
    ("sm_103a", 6144, 128, "bf16"): ((512, 8192),),
    ("sm_103a", 6144, 128, "f32"): ((512, 8192),),
}
#: The measured token counts of the 148-SM audit.
K1_148_MEASURED_TOKENS = (512, 1024, 2048, 4096, 8192, 16231, 32768, 65536)
#: Token counts of the 148-SM entries measured for the FP64 error only (one token outside a measured entry).
K1_148_ERROR_TOKENS = (32767, 32769, 65537)


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
        {T + d for T in K1_148_MEASURED_TOKENS for d in (-1, 0, 1)}
        | {T + d for T in K1_148_ERROR_TOKENS for d in (-1, 0, 1)}
        | {64, 256, 131072}
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
            ends = K1_148_MEASURED_TOKENS + K1_148_ERROR_TOKENS
            for lo, hi in spans:
                assert lo in ends, key
                assert hi is None or hi in ends, key
                assert without[(key, 148, lo)] is not None, key
                assert with_table[(key, 148, lo)] is None, key


#: The 216-SM entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` at single measured token counts outside 16172 to 16384
#: tokens, ``key -> token counts``: from 216 SMs up these rows run the deterministic program there.
K1_216_SINGLE_TOKENS = {
    ("sm_107a", 1, 4096, 32, "bf16"): (512, 8192),
    ("sm_107a", 1, 4096, 48, "f32"): (512,),
    ("sm_107a", 1, 4096, 64, "f32"): (512,),
    ("sm_107a", 1, 5120, 32, "bf16"): (512,),
    ("sm_107a", 1, 5120, 32, "f32"): (512,),
    ("sm_107a", 1, 5120, 48, "f32"): (512,),
    ("sm_107a", 1, 5120, 64, "f32"): (512,),
    ("sm_107a", 1, 5120, 1024, "bf16"): (65536,),
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


def test_k1_216_sm_routing_is_the_record_table():
    """On 216-SM sm_107a parts every row decides by :data:`K1_SPLIT_RULES` (the record verdicts), except the rows of
    the 216-SM table of :data:`K1_SPLIT_OVERRIDES_BY_SM` at the token counts of :data:`K1_SPLIT_RULES_BY_SM_TOKENS`,
    and except the 216-SM token counts of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`, which run the deterministic
    program."""
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
    overrides = cake_backend.K1_SPLIT_OVERRIDES_BY_SM[("sm_107a", 216)]
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
    monkeypatch,
):
    """``4096 -> 1024`` FP32 takes the 192-wide row-major program at one split on exactly 216 SMs at 16172 to 16384
    tokens (:data:`K1_SPLIT_OVERRIDES_BY_SM`), except at 16231 tokens (its 216-SM entry of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`: the deterministic program); one token outside that window and on 217 SMs
    it keeps its :data:`K1_SPLIT_RULES` verdict (256-wide tiles at one split), on 215 SMs (no table) it runs the
    deterministic program, and on 212 SMs its own table decides.  On 216 SMs the launch is the row-major-store
    program's: 88 tiles, 176 CTAs."""
    arch, d_in, d_out = "sm_107a", 4096, 1024
    key = (arch, 1, d_in, d_out, "f32")
    overrides = cake_backend.K1_SPLIT_OVERRIDES_BY_SM
    assert overrides == {("sm_107a", 216): {key: (192, 1)}}
    assert cake_backend.K1_SPLIT_RULES[key] == (256, 1)

    def plan(sms, T):
        return cake_backend.k1_plan(
            arch=arch, L=1, M=d_in, N=d_out, K=T, out_f32=True, sm_count=sms
        )

    for T in (16172, 16200, 16300, 16384):
        assert plan(216, T) == (192, 1), T
        assert plan(217, T) == (256, 1), T
        assert plan(212, T) == (192, 1), T
        assert plan(215, T) is None, T
    # 16231 tokens: the deterministic program from 216 SMs up and on 208 / 212 SMs
    for sms in (208, 212, 215, 216, 217, 280):
        assert plan(sms, 16231) is None, sms
    for T in (1024, 8192, 16171, 16385, 32768):
        assert plan(216, T) == (256, 1), T
        assert plan(217, T) == (256, 1), T
        assert plan(212, T) is None, T
    # every other row and token count on 216 SMs: the record verdict; BF16 4096 -> 1024 has none
    assert (
        cake_backend.k1_plan(
            arch=arch, L=1, M=d_in, N=d_out, K=16384, out_f32=False, sm_count=216
        )
        is None
    )
    # the planned launch on 216 SMs
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
    outside = cake_backend.plan_atomic_gemm(
        X[:16171].t(), G[:16171], out, transposed_out=True, arch=arch, sm_count=216
    )[0]
    assert (outside.template, outside.block_n, outside.out_t) == (
        "dense_proj_gemm_atomic_nn_n256_f32_t",
        256,
        True,
    )
    # an override entry of None keeps the row on the deterministic program in the window only; a table of another
    # SM count does not act on 216 SMs
    monkeypatch.setattr(
        cake_backend,
        "K1_SPLIT_OVERRIDES_BY_SM",
        {(arch, 216): {key: None}, (arch, 220): {key: (128, 2)}},
    )
    assert plan(216, 16384) is None and plan(216, 16385) == (256, 1)
    assert plan(220, 16384) == (128, 2) and plan(221, 16384) == (256, 1)


def test_k1_sm107a_token_counts_of_the_deterministic_rules():
    """The sm_107a entries of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS` at 16172 to 16384 tokens are the single token
    counts 16231 and 16384 per SM count: an entry of 208 or 212 SMs holds on exactly that SM count (not on the other
    table's part, nor from 216 SMs up), an entry of 216 SMs from 216 SMs up; the neighbouring token counts keep the
    row's verdict.  (The 216-SM ranges of the four wide FP32 rows at large token counts: the next test.)"""

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
#: parts of 216 SMs or more: their one-split program met the FP32 error budget on every seed at 40960 tokens and
#: missed it from 45056 tokens up.
K1_SM107A_FP32_BUDGET_TOKENS = 40961
#: Those rows ``(d_in, d_out)`` and their one-split programs ``(block_n, splits)`` below that token count.
K1_SM107A_WIDE_FP32_ROWS = {
    (4096, 512): (128, 1),
    (4096, 1024): (256, 1),
    (5120, 512): (128, 1),
    (5120, 1024): (256, 1),
}


def test_k1_sm107a_wide_fp32_rows_at_large_token_counts():
    """The one-split programs of the sm_107a FP32 rows ``4096 -> 512``, ``4096 -> 1024``, ``5120 -> 512`` and
    ``5120 -> 1024`` miss the deterministic package's FP32 error budget at large token counts: from
    ``K1_SM107A_FP32_BUDGET_TOKENS`` up, with no upper bound, these rows run the deterministic program on parts of 216
    SMs or more (their entries in :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS`) and keep their one-split program one token
    below.  The 208-SM and 212-SM tables hold only at 16172 to 16384 tokens, so those parts run the deterministic
    program at these token counts as before; the other entries of these rows lie in that window.  No other sm_107a
    row has such a range, and sm_100a / sm_103a have no verdict for these rows."""
    T0 = K1_SM107A_FP32_BUDGET_TOKENS
    rows = K1_SM107A_WIDE_FP32_ROWS
    excluded = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS
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
            for T in (32768, T0 - 1):
                assert p(M, N, sms, T) == choice, (M, N, sms, T)
            for T in (T0, 45056, 65536, 65537, 131072):
                assert p(M, N, sms, T) is None, (M, N, sms, T)
        for sms in (208, 212, 215):
            for T in (32768, T0 - 1, T0, 65536, 131072):
                assert p(M, N, sms, T) is None, (M, N, sms, T)
        for arch, sms in (("sm_100a", 148), ("sm_100a", 152), ("sm_103a", 152)):
            assert (arch, 1, M, N, "f32") not in cake_backend.K1_SPLIT_RULES
            for T in (T0 - 1, T0, 65536):
                assert p(M, N, sms, T, arch=arch) is None, (arch, M, N, T)


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


def test_k1_row_major_verdict_runs_the_row_major_program(monkeypatch):
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
    # weight gradient at T = 16231 (508 K blocks), inside the token windows (test_router_wgrad_token_windows):
    # sm_100a hidden 4096 "H1x5p44" (44 head units, then 5 tail units per CTA pair), hidden 5120 "H0.5x3" on sm_100a /
    # sm_103a (76 head + 228 tail units) and "H0.5x4" on sm_107a (108 head + 432 tail units on 216 SMs; 106 + 424 on
    # 212 and 104 + 416 on 208, ROUTER_RULES_BY_SM); hidden 6144, sm_103a hidden 4096 and sm_107a hidden 4096: the
    # deterministic plan
    for arch in ("sm_100a", "sm_103a"):
        assert cake_backend.ROUTER_RULES[(arch, "nn_t", 6144, 256)] is None
        assert (
            cake_backend.router_units(
                "nn_t", arch=arch, sm_count=152, M=6144, N=256, K=16231
            )
            is None
        )
    assert cake_backend.ROUTER_RULES[("sm_100a", "nn_t", 4096, 256)] == (
        (10240, 16384, "H1x5p44"),
    )
    assert cake_backend.router_partition(
        "nn_t", arch="sm_100a", sm_count=152, M=4096, N=256, K=16231
    ) == (44 + 5 * 76, 22, round(16 * 508 * 44 / 76))
    assert cake_backend.ROUTER_RULES[("sm_103a", "nn_t", 4096, 256)] is None
    assert (
        cake_backend.router_partition(
            "nn_t", arch="sm_103a", sm_count=152, M=4096, N=256, K=16231
        )
        is None
    )
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
        for sms in (208, 212, 216):
            assert (
                cake_backend.router_partition(
                    "nn_t", arch="sm_107a", sm_count=sms, M=hidden, N=256, K=16231
                )
                is None
            )
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
    # few tokens at hidden 5120: the "waves" partition (100 tokens, 4 K blocks: three waves of the 76 CTA pairs, at
    # most one unit per (tile, K block) step -> 160 units)
    for arch in ("sm_100a", "sm_103a"):
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=100
        ) == (160, 0, 0)
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
# other token count runs the deterministic program
ROUTER_WGRAD_WINDOWS = {
    ("sm_100a", 4096): ((10240, 16384, "H1x5p44"),),
    ("sm_100a", 5120): ((1, 128, "waves"), (10240, 32768, "H0.5x3")),
    ("sm_103a", 5120): ((1, 192, "waves"), (16172, 32768, "H0.5x3")),
    ("sm_107a", 5120): ((16172, 16384, "H0.5x4"),),
}
# the parts with a router table of their own (ROUTER_RULES_BY_SM) and their weight-gradient windows,
# (arch, sm_count, hidden) -> ((first, last, spec), ...)
ROUTER_BY_SM_PARTS = {("sm_107a", 208), ("sm_107a", 212)}
ROUTER_WGRAD_WINDOWS_BY_SM = {
    ("sm_107a", 208, 5120): ((16172, 16384, "H0.5x4"),),
    ("sm_107a", 212, 5120): ((16172, 16384, "H0.5x4"),),
}


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_router_wgrad_token_windows(arch):
    """The weight gradient runs a partition only inside its token windows: at the first and the last token count of a
    window, not one token outside it, not above 32768 tokens, and around the few-token "waves" windows; a head/tail
    window only on the SM count of its table (:data:`ROUTER_SM_COUNT` for :data:`ROUTER_RULES`, the part's own for
    :data:`ROUTER_RULES_BY_SM`; not one SM more or fewer), a "waves" window on every SM count where
    :func:`router_waves` gives a partition.  The deterministic program at every other token count and SM count and
    on every hidden size without a window."""
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
                spec = next((s for a, b, s in windows if a <= T <= b), None)
                k_iters = -(-T // 32)
                got = cake_backend.router_partition(
                    "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=T
                )
                if spec == "waves":
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
    if arch == "sm_100a":
        # the first token count of the hidden 4096 window (320 K blocks) and the smallest measured "waves" count
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=152, M=4096, N=256, K=10240
        ) == (44 + 5 * 76, 22, round(16 * 320 * 44 / 76))
        assert cake_backend.router_partition(
            "nn_t", arch=arch, sm_count=152, M=5120, N=256, K=64
        ) == (80, 0, 0)


def test_router_per_sm_count_tables_decide_only_their_own_sm_count(monkeypatch):
    """A part whose SM count has a table in :data:`ROUTER_RULES_BY_SM` takes that table only (every row of
    :data:`ROUTER_RULES` of its architecture has an entry there, no ``"waves"`` window, closed windows); one SM more or
    fewer runs the deterministic program, and the measured SM count of :data:`ROUTER_RULES` decides as before."""
    for (arch, sms), table in cake_backend.ROUTER_RULES_BY_SM.items():
        assert sms != cake_backend.ROUTER_SM_COUNT[arch]
        assert set(table) == {k for k in cake_backend.ROUTER_RULES if k[0] == arch}
        for (key_arch, layout, _hidden, _experts), rule in table.items():
            assert key_arch == arch
            if rule is None:
                continue
            assert layout == "nn_t" and isinstance(rule, tuple)
            for first, last, spec in rule:
                assert 1 <= first <= last <= 32768
                assert cake_backend.router_head_tail(spec) is not None
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
    # 212 and 208 SMs: the tables of their own (ROUTER_RULES_BY_SM), hidden 5120 "H0.5x4" on 106 / 104 CTA pairs
    for sms, pairs in ((212, 106), (208, 104)):
        plan, _ = cake_backend.plan_atomic_router(
            G.t(), X5, torch.zeros(N, 5120), arch=arch, sm_count=sms
        )
        assert plan.template == "router_fp32_gemm_atomic_nn_t_ht_c3_nreg"
        assert (plan.units, plan.head_spans, plan.head_steps) == (
            5 * pairs,
            pairs // 2,
            20 * 254,
        )
        assert plan.grid == (10 * pairs, 1, 1)
        X4 = torch.zeros(T, 4096)
        plan, reason = cake_backend.plan_atomic_router(
            G.t(), X4, torch.zeros(N, 4096), arch=arch, sm_count=sms
        )
        assert plan is None and "planner" in reason
    # sm_100a / sm_103a, hidden 5120: three tail units per pair (76 head + 228 tail units on 152 SMs); sm_103a runs
    # the head/tail program with the A parts in tensor memory, and the deterministic program while this checkout does
    # not register it
    ts = "router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg"
    templates = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES
    assert templates == {"sm_103a": ts}
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
    """With this checkout's registry and router tables, on every architecture and the SM counts of the measured parts,
    the weight gradient (256 experts, hidden 4096 / 5120 / 6144, token counts at and around the windows) runs the
    head/tail program with the A parts in tensor memory exactly on sm_103a with 152 SMs, hidden 5120 and 16172 to
    32768 tokens (the sm_103a head/tail window of ``ROUTER_RULES``); every other head/tail plan runs
    ``router_fp32_gemm_atomic_nn_t_ht_c3_nreg``, and both carry the planner's partition.  The program is registered
    for sm_103a only."""
    ts = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES["sm_103a"]
    c3 = "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX
    assert [a for a, table in cake_jit.KERNELS.items() if ts in table] == ["sm_103a"]
    assert cake_jit.MODULES[cake_jit.select_module("sm_103a", ts)]["arches"] == [
        "sm_103a"
    ]
    N, seen = 256, {}
    for arch in cake_jit.ARCHES:
        for sms in ROUTER_SM_FACTS:
            for hidden in (4096, 5120, 6144):
                out = torch.empty(N, hidden)
                for T in (8192, 10240, 16171, 16172, 16231, 16384, 16385, 32768, 32769):
                    # stride-0 views of the gradient [T, N] and the activations [T, hidden]: the layout classes of
                    # G.t() and X without their memory
                    G = torch.empty(1, N).expand(T, N)
                    X = torch.empty(1, hidden).expand(T, hidden)
                    plan, _ = cake_backend.plan_atomic_router(
                        G.t(), X, out, arch=arch, sm_count=sms
                    )
                    in_window = (
                        arch == "sm_103a"
                        and sms == 152
                        and hidden == 5120
                        and 16172 <= T <= 32768
                    )
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
    # 16172, 16231, 16384, 16385 and 32768 tokens
    assert seen.pop(("sm_103a", 152, ts)) == 5
    assert seen and all(template == c3 for _, _, template in seen)
    assert {(arch, sms) for arch, sms, _ in seen} == {
        ("sm_100a", 152),
        ("sm_107a", 208),
        ("sm_107a", 212),
        ("sm_107a", 216),
    }


def test_k2_values_feed_every_registered_router_program():
    """The keyword arguments of a K2 launch name every argument of every registered router program's argument plan
    (the head/tail program's ``head_spans`` / ``head_steps`` included)."""
    out = torch.zeros(256, 5120)
    plan = cake_backend.AtomicRouterPlan(
        M=5120, N=256, K=16231, a_mn=True, b_mn=True, swapped=True, layout="nn_t", m_tiles=40,
        n_tiles=2, k_iters=508, units=540, template="router_fp32_gemm_atomic_nn_t_ht_c3_nreg",
        head_spans=54, head_steps=5080,
    )  # fmt: skip
    values = cake_backend._k2_values(plan, A="a", parts="b", out=out)
    names = [
        n for n, r in cake_jit.MODULES.items() if r["template"].startswith("router_")
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
    and of this package's plan whenever the plan has the entry point's flags, so the launch binds the same
    arguments; contiguous operands always have them, other layouts fall back to ``_k2_views``."""
    arch, sm_count, T, H = "sm_103a", 152, 64, 5120
    _register(
        monkeypatch, arch, "router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"
    )
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
        if op != "dgrad":
            plan, _ = cake_backend.plan_atomic_router(
                A, B, out, arch=arch, sm_count=sm_count, splits=2
            )
            if plan is not None:
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
    PDL flag is bound into both lists."""
    monkeypatch.setattr(cake_backend.tvm_ffi, "from_dlpack", lambda t, **_: t)
    arch, T, H = "sm_103a", 64, 6144
    monkeypatch.setattr(det_backend, "require_arch", lambda device: arch)
    monkeypatch.setattr(
        det_backend,
        "load_cake_dense_projection_det_gemm_module",
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
        else:
            assert bound.partials_shape is None and bound.workspace is None
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


def test_k1_sm107a_wide_fp32_rows_bind_per_token_count(monkeypatch):
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


def test_router_rule_spec_is_the_planners_verdict():
    """Where ``_router_rule_spec`` gives no spec on an SM count, ``router_partition`` without a split count runs the
    deterministic program there (the convenience path's pre-check then skips the planner exactly where the planner
    would keep the row); a spec it gives is the planner's (a head/tail or unit-count spec is the partition on an SM
    count whose router table holds, ``router_rules``)."""
    rows = dict(cake_backend.ROUTER_RULES)
    for arch in cake_backend.ROUTER_SM_COUNT:
        for layout in ("kk", "nn_t"):
            rows.setdefault((arch, layout, 2048, 256), None)  # rows without an entry
            rows.setdefault((arch, layout, 5120, 512), None)
    checked = 0
    for (arch, layout, hidden, experts), rule in rows.items():
        for T in _router_spec_tokens(rule):
            if layout == "kk":
                M, N, K = T, experts, hidden
            else:
                M, N, K = hidden, experts, T
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
    """``_k2_row_without_verdict`` is true exactly for the valid ``kk`` / ``nn_t`` bindings the planner (no split count)
    keeps on the deterministic program for want of a verdict, and false for every binding the planner refuses, plans
    or keeps for another reason (left to the planner); a verdict added at run time is seen."""
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
    assert not without_verdict(
        torch.empty(64, 256), torch.empty(256, 5120), torch.empty(64, 5120), arch, 152
    )  # the kn class (the input gradient)
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
    G, X = torch.empty(64, 256), torch.empty(64, 6144)
    geometry = cake_backend._k2_geometry(G.t(), X, torch.empty(256, 6144))
    assert runs(geometry, 1) and runs(geometry, 0)
    own = cake_backend.ROUTER_RULES_BY_SM[("sm_107a", 212)]
    monkeypatch.setitem(own, ("sm_107a", "nn_t", 6144, 256), ((1, 64, "pairs"),))
    assert not runs(geometry, 1) and runs(geometry, 0) and runs(geometry, 2)
    own[("sm_107a", "nn_t", 6144, 256)] = None
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
                M, N, K = (
                    (T, experts, hidden) if layout == "kk" else (hidden, experts, T)
                )
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
    # the sm_107a hidden-5120 weight gradient: "H0.5x4" at 16172 to 16384 tokens on 208, 212 and 216 SMs
    for sms in (208, 212, 216):
        for T, want in ((16171, None), (16172, "H0.5x4"), (16384, "H0.5x4")):
            assert (
                cake_backend._router_rule_spec(
                    "nn_t", arch="sm_107a", sm_count=sms, M=5120, N=256, K=T
                )
                == want
            )
    # a verdict in the 212-SM table only, then in ROUTER_RULES only (216 SMs): the no-verdict check of a binding of
    # the row follows the table of each SM count
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
        "k2_kn": "det",
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
    (``dense_proj_gemm_atomic_nn_n<64|128|256>[_f32]_t`` and the row-major ``dense_proj_gemm_atomic_nn_n192[_f32]``),
    the two router classes and the weight gradient's head/tail program."""
    wanted = {
        cake_backend.atomic_k1_template(bn, f32)
        for bn in cake_backend.K1_BLOCK_N_CHOICES
        for f32 in (False, True)
    } | {
        "router_fp32_gemm_atomic_kk",
        "router_fp32_gemm_atomic_nn_t",
        "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX,
    }
    for arch, table in cake_jit.KERNELS.items():
        assert set(table) <= wanted | {cake_backend.router_head_tail_template(arch)}, (
            sorted(set(table) - wanted)
        )


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
    """The deterministic package's K1 budgets against FP64."""
    assert bool(torch.isfinite(out).all()), what
    if out.dtype == torch.bfloat16:
        bad = (out.double() - ref).abs() > BF16_ATOL + BF16_RTOL * ref.abs()
        assert int(bad.sum()) == 0, (what, int(bad.sum()))
    else:
        assert _rel_fro(out, ref) <= F32_REL_FRO_MAX, (what, _rel_fro(out, ref))
        assert float(
            (out.double() - ref).abs().max()
        ) <= F32_MAX_ABS_OVER_REF_ABSMAX * float(ref.abs().max()), what


def _k1_bf16_misses(out, ref):
    """Elements of a BF16 output outside the deterministic package's BF16 budget against FP64."""
    return int(((out.double() - ref).abs() > BF16_ATOL + BF16_RTOL * ref.abs()).sum())


def _k1_bf16_budget_ratio(out, ref):
    """The largest error of a BF16 output against FP64 over its BF16 budget (above 1: outside the budget)."""
    err = (out.double() - ref).abs() / (BF16_ATOL + BF16_RTOL * ref.abs())
    return float(err.max())


def _k1_check_or_as_det(out, G, X, what=""):
    """:func:`_k1_check`, except where the deterministic program misses the BF16 budget on the same inputs too (FP32
    accumulation over the largest token counts): then a relative Frobenius error and a BF16 budget ratio
    (:func:`_k1_bf16_budget_ratio`) each within 2 % of that program's."""
    ref = G.double().t() @ X.double()
    if out.dtype != torch.bfloat16 or _k1_bf16_misses(out, ref) == 0:
        _k1_check(out, ref, what)
        return
    det = torch.full_like(out, float("nan"))
    det_backend.projection_det_wgrad(G, X, det)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all()), what
    assert _k1_bf16_misses(det, ref) > 0, (what, _k1_bf16_misses(out, ref))
    rf, rf_det = _rel_fro(out, ref), _rel_fro(det, ref)
    br, br_det = _k1_bf16_budget_ratio(out, ref), _k1_bf16_budget_ratio(det, ref)
    assert rf <= 1.02 * rf_det, (what, rf, rf_det)
    assert br <= 1.02 * br_det, (what, br, br_det)


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
    """``k1_rule(G, X, out, splits)``: a verdict for the row (and no SM-count floor, excluded token counts or tables
    of exactly one SM count), so the convenience path runs this package's program at that split count; returns the
    template."""

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
        monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (block_n, splits))
        monkeypatch.setattr(cake_backend, "K1_SPLIT_OVERRIDES_BY_SM", {})
        monkeypatch.setattr(cake_backend, "K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM", {})
        monkeypatch.setattr(cake_backend, "K1_SPLIT_RANGE_OVERRIDES_BY_SM", {})
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
    """``(A, B, out)`` of the router GEMM of ``op`` (``fwd``: ``X @ W.T``, ``wgrad``: ``G.T @ X``)."""
    if op == "fwd":
        return (
            X,
            W.t(),
            torch.full((X.shape[0], W.shape[0]), float("nan"), device=X.device),
        )
    return G.t(), X, torch.full((G.shape[1], X.shape[1]), float("nan"), device=X.device)


@pytest.fixture
def router_rule(monkeypatch):
    """``router_rule(A, B, spec)``: a verdict for the router row of ``(A, B)``, so the default plan runs this
    package's program with the partition ``spec`` (or the deterministic program with ``None``)."""

    def force(op, A, B, spec):
        arch = det_backend.arch_for(A.device)
        if op == "fwd":
            key = (arch, "kk", int(A.shape[1]), int(B.shape[1]))
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


@pytest.mark.parametrize("T", [16172, 16231, 32768])
def test_gpu_router_tmem_head_tail_program_on_152_sm_sm_103a(T):
    """On a 152-SM sm_103a part the default plan of the hidden-5120 weight gradient in its head/tail window runs the
    head/tail program with the A parts in tensor memory (``ROUTER_HEAD_TAIL_TEMPLATES``) and meets the gate of
    :func:`test_gpu_router_d11_gate` over ``RERUNS`` launches into a NaN-poisoned ``out``: finite, within the
    reordering bound of the K-order sum of the deterministic program's one-split partials of its segments, relative
    Frobenius error against FP64 at most ``ROUTER_REL_FRO_MAX``.  On whole tiles (one unit per tile) it stores the bits
    of the three-chunk-buffer head/tail program at the same partition, which are the deterministic program's one-split
    result.  Skips on any other device."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    if arch != "sm_103a" or sm_count != 152:
        pytest.skip(f"needs a 152-SM sm_103a part, got {arch} with {sm_count} SMs")
    ts = cake_backend.ROUTER_HEAD_TAIL_TEMPLATES["sm_103a"]
    c3 = "router_fp32_gemm_atomic_nn_t" + cake_backend.ROUTER_HEAD_TAIL_SUFFIX
    _require_template(arch, ts)
    _require_template(arch, c3)
    X, W, G = _router_inputs(T, device, SEED + 30, K=5120)
    A, B, out = _router_case("wgrad", X, W, G)
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
        A, B, out = _router_case("fwd" if layout == "kk" else "wgrad", X, W, G)
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
    program at the first and the last token count of :data:`K1_SPLIT_RULES_BY_SM_TOKENS` (``4096 -> 1024`` FP32 at
    16172 and 16384 tokens: the 192-wide row-major-store program at one split), meets the deterministic package's
    FP64 budget and, where both run one split, gives the bits of the row's :data:`K1_SPLIT_RULES` program (256-wide
    transposed tiles) on the same inputs.  A part of any other SM count skips."""
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
            want = cake_backend.k1_plan(
                arch=arch, L=L, M=M, N=N, K=T, out_f32=f32, sm_count=sm_count
            )
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
    :data:`K1_SPLIT_RULES_BY_SM_TOKENS` and gives the bits of the prepared launch, the tickets zero after.  A part of
    any other SM count skips."""
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
            plan = cake_backend.k1_plan(
                arch=arch, L=L, M=M, N=N, K=T, out_f32=f32, sm_count=sm_count
            )
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
    """``(key, choice, T, want)`` for every row of the part's ranges in :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and
    :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM`: the first and the last token count of each range (``want``: the program of
    the part's table, or of the override range) and one token outside it where neither the window nor another range of
    the row holds (``want`` ``None``: the deterministic program); ``choice`` is the planner's."""
    first, last = cake_backend.K1_SPLIT_RULES_BY_SM_TOKENS
    part = (arch, sm_count)
    rows = cake_backend.K1_SPLIT_RULES_BY_SM_TOKEN_RANGES.get(part, {})
    over = cake_backend.K1_SPLIT_RANGE_OVERRIDES_BY_SM.get(part, {})
    table = cake_backend.K1_SPLIT_RULES_BY_SM.get(part, {})
    cases = []
    for key in sorted(set(rows) | set(over)):
        _, L, M, N, dt = key
        spans = [(lo, hi, table[key]) for lo, hi in rows.get(key, ())]
        spans += [(lo, hi, v) for lo, hi, v in over.get(key, ())]
        for lo, hi, _ in spans:
            for T in sorted({lo, hi, lo - 1, hi + 1}):
                held = [v for a, b, v in spans if a <= T <= b]
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
    (exactly ``sm_count``): the first and the last token count of each range (65537 for a range without an upper bound;
    ``want`` ``None``: the deterministic program) and one token outside it (``want``: the row's :data:`K1_SPLIT_RULES`
    verdict); ``choice`` is the planner's."""
    rows = cake_backend.K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM.get((arch, sm_count), {})
    cases = []
    for key, spans in sorted(rows.items()):
        _, L, M, N, dt = key
        rule = cake_backend.K1_SPLIT_RULES[key]
        held = lambda T, spans=spans: any(  # noqa: E731
            lo <= T and (hi is None or hi >= T) for lo, hi in spans
        )
        for lo, hi in spans:
            ends = {lo, 65537 if hi is None else hi}
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


def test_gpu_k1_sm_count_token_ranges_run_at_their_edges():
    """Unforced (``deterministic=False``, no split count), on a part of exactly the SM count of a table of
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` or :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` (``sm_107a``: 212 SMs): at the
    first and the last token count of every range of a row the prepared launch runs the program of the part's table
    in :data:`K1_SPLIT_RULES_BY_SM` (of the range, for an override range) and meets the deterministic package's FP64
    budget (:func:`_k1_check_or_as_det`); one token outside a range (and outside the window) the row runs the
    deterministic program, with the deterministic package's bits.  On a part of exactly the SM count of a table of
    :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` (148-SM ``sm_100a`` / ``sm_103a`` parts) the reverse: the deterministic
    program with its bits at both ends of a range, the row's program one token outside it.  A part of any other SM
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
        G, X = _bf16((T, N), device, g), _bf16((T, M), device, g)
        out_dtype = torch.float32 if f32 else torch.bfloat16
        out = torch.full((N, M), float("nan"), device=device, dtype=out_dtype)
        prepared = cake_backend.prepare_projection_wgrad(G, X, out, deterministic=False)
        if want_plan is not None:
            assert plan == want_plan, (key, T)
            assert isinstance(prepared, cake_backend.PreparedAtomicGemm), (key, T)
            assert (prepared.plan.block_n, prepared.plan.splits) == plan, (key, T)
            _assert_ran(prepared.launch, prepared.module_name)
            torch.cuda.synchronize()
            _k1_check_or_as_det(out, G, X, f"{M} {N} {dt} {T}")
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
    :data:`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` or :data:`K1_SPLIT_RANGE_OVERRIDES_BY_SM` (``sm_107a``: 212 SMs): the
    convenience path binds the part's program (the range's, for an override range) at the first and the last token
    count of every range of a row (its :data:`_K1_ATOMIC_CACHE` entry) and gives the
    bits of the prepared launch where at most two partials meet (the FP64 budget otherwise), the tickets zero after;
    one token outside a range it binds the deterministic package's plan.  On a part of exactly the SM count of a table
    of :data:`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` the reverse.  A part of any other SM count skips."""
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
    at one token) the prepared launch runs the partition the planner gives -- a head/tail window on the measured SM
    count, a "waves" window where :func:`cake_backend.router_waves` gives a partition -- and meets the gate; one
    token outside the window, or on another SM count, it runs the deterministic launch."""
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
                prepared = cake_backend.prepare_router_fp32_gemm(
                    A, B, out, deterministic=False
                )
                if not inside:
                    assert part is None, (hidden, T)
                elif spec != "waves" and measured:
                    assert part is not None, (hidden, T)
                if part is None:
                    assert not isinstance(prepared, atomic), (hidden, T)
                    continue
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


def test_gpu_router_verdicts_set_at_run_time_are_seen(monkeypatch):
    """The convenience path keeps whether a router row has a verdict (per device and view geometry) and whether any
    router table gives the forward one until a router table changes: on a row without a verdict the forward and the
    weight gradient give the deterministic program's bits (unplanned, also on the second call, from the memo); a
    verdict set at run time in the table of the device's SM count runs this package's program at the next call,
    without clearing a cache, and its removal the deterministic program again."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
    table, _ = cake_backend.router_rules(arch, sm_count)
    if table is cake_backend.ROUTER_RULES:
        # a verdict of ROUTER_RULES applies on this device whatever its SM count
        monkeypatch.setitem(cake_backend.ROUTER_SM_COUNT, arch, sm_count)
    hidden, T = 6144, 1000  # no verdict on any architecture
    X, W, G = _router_inputs(T, device, SEED + 26, K=hidden)
    keys = {"fwd": (arch, "kk", hidden, 256), "wgrad": (arch, "nn_t", hidden, 256)}
    assert all(table.get(k) is None for k in keys.values())
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


def test_gpu_router_deterministic_program_through_the_binding():
    """``deterministic=False`` calls that run the deterministic program -- the forward and the input gradient, the
    weight gradient of a row without a verdict, the generic ``kn`` call -- go through this package's binding
    (``_K2_SPLITK_CACHE``; nothing in ``_K2_ATOMIC_CACHE``) and give the deterministic package's bits: dense and
    padded outputs (one split stores into a padded ``out`` through its flat alias, several splits reduce into it), a
    split source with padded rows (the torch chain), repeated calls; a larger row grows the arena and the smaller
    rows' launches are carved again with the same bits; another stream binds an arena of its own; a binding the
    deterministic package refuses raises its error."""
    device, arch = _atomic_device_or_skip()
    hidden = 6144  # no router verdict on any architecture
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

    variants = ((64, False, False), (1000, True, False), (1000, False, True))
    for T, pad_out, pad_g in variants:
        check(T, SEED + 70 + T, pad_out, pad_g)
    assert cake_backend._K2_SPLITK_CACHE and not cake_backend._K2_ATOMIC_CACHE
    entries = list(cake_backend._K2_SPLITK_CACHE.values())
    assert all(e.k2 is not None for e in entries)
    assert any(e.k2.partials_shape is not None for e in entries)
    assert any(e.k2.partials_shape is None for e in entries)
    assert any(e.split_entry is None for e in entries)  # the padded split source
    versions = {id(e): e.k2.version for e in entries}
    check(8000, SEED + 71)  # a larger row: the arena grows
    check(64, SEED + 70 + 64)
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


def test_gpu_router_binding_pdl_waits_for_an_early_producer():
    """The deterministic router program through this package's binding as programmatic dependent launches: the
    forward writes ``Y``, the input gradient reads it and writes ``dX``, the weight gradient (a row without a verdict)
    reads both; each starts while its predecessor runs and must still read every value it wrote.  The flag is bound
    into the cached argument lists (a key per flag), and every launch gives the deterministic package's bits."""
    device, arch = _atomic_device_or_skip()
    hidden, T = 6144, 4096  # no router verdict on any architecture
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


def test_gpu_cached_router_launch_runs_the_head_tail_windows_at_their_edges():
    """Unforced (``deterministic=False``, no split count): at the first and the last token count of every head/tail
    window of this device's router table (``router_rules`` of its architecture and SM count), the convenience path's
    cached launch (its bound arguments, ``_K2Binding``) runs the prepared launch's plan and program, meets the gate
    and, for a reproducible plan, gives the prepared launch's bits; the second call of the key reuses the binding."""
    device, arch = _atomic_device_or_skip()
    sm_count = det_backend.device_facts(device).sm_count
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
