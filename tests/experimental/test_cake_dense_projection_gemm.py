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
import inspect
import json
import re
import sys
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
    "dense_projection_gemm": (("A", "B", "out"), {"transposed_out": True}, (("A", "B", "out"), {"transposed_out": True})),
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
        assert kwargs["det_call"]() is det.result  # a row the planner does not take
        assert det.calls == [((A, B, out), {"transposed_out": False})]
    else:
        assert result is det.result and not atomic.calls
        assert det.calls == [((A, B, out), {"transposed_out": False})]


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


def test_k2_routes(monkeypatch):
    """``deterministic=False``: the router forward and weight gradient reach this package's stream-K path; the
    input gradient and the split stage run the deterministic programs."""
    dgrad, split = (
        _patch_det(monkeypatch, "router_dgrad"),
        _patch_det(monkeypatch, "split_fp32_to_bf16x3_"),
    )
    gemm = _patch_det(monkeypatch, "router_fp32_gemm")
    atomic = Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic)
    T, K, N = 64, 32, 16
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    assert cake_backend.router_forward(X, W, deterministic=False) is atomic.result
    assert (
        cake_backend.router_wgrad(G, X, deterministic=False, splits=5) is atomic.result
    )
    assert cake_backend.router_dgrad(G, W, deterministic=False) is dgrad.result
    assert (
        cake_backend.split_fp32_to_bf16x3_("s", "p", "r", deterministic=False)
        is split.result
    )
    (fwd_args, fwd_kwargs), (wg_args, wg_kwargs) = atomic.calls
    assert (
        fwd_args[0] == "launch"
        and fwd_args[1] is X
        and fwd_args[2].equal(W.t())
        and fwd_kwargs == {"splits": None}
    )
    assert tuple(fwd_args[3].shape) == (T, N) and fwd_args[3].dtype == torch.float32
    assert wg_args[1].equal(G.t()) and wg_args[2] is X and wg_kwargs == {"splits": 5}
    assert tuple(wg_args[3].shape) == (N, K)
    # the generic entry: the kn class runs the deterministic program, kk / nn_t this package's
    assert (
        cake_backend.router_fp32_gemm(
            G, W, torch.zeros(T, K), deterministic=False, splits=1
        )
        is gemm.result
    )
    assert (
        cake_backend.router_fp32_gemm(X, W.t(), torch.zeros(T, N), deterministic=False)
        is atomic.result
    )
    assert (
        cake_backend.router_fp32_gemm(G.t(), X, torch.zeros(N, K), deterministic=False)
        is atomic.result
    )
    assert len(gemm.calls) == 1 and len(atomic.calls) == 4


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
        ("sm_107a", 216, 34),
    ],
)
def test_k1_planners_take_the_measured_rows(arch, sm_count, rows):
    """The verdicts of the three architectures take 28 / 32 / 34 of the 124 weight-gradient rows (the MLA head rows,
    ``L = 64``, run the deterministic program), at token counts near 16k; on 148 SMs sm_103a leaves ``4096 -> 48`` to
    the deterministic program (28 rows)."""
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
    # sm_107a: from 216 SMs only
    assert cake_backend.k1_plan(arch="sm_107a", M=4096, N=32, sm_count=216, **kw) == (
        64,
        4,
    )
    assert (
        cake_backend.k1_plan(arch="sm_107a", M=4096, N=32, sm_count=212, **kw) is None
    )
    assert (
        cake_backend.k1_plan(arch="sm_107a", M=4096, N=48, sm_count=216, **kw) is None
    )
    assert cake_backend.k1_plan(
        arch="sm_107a", M=4096, N=512, sm_count=216, L=1, K=16231, out_f32=True
    ) == (128, 1)
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


def test_k1_small_token_counts_run_the_deterministic_program():
    """``num_splits * 64 <= T + 63``: a token count too short for the split count runs the deterministic program."""
    kw = dict(arch="sm_107a", L=1, M=4096, N=32, out_f32=False, sm_count=216)
    assert cake_backend.k1_plan(K=193, **kw) == (64, 4)
    assert cake_backend.k1_plan(K=192, **kw) is None
    kw = dict(arch="sm_103a", L=1, M=4096, N=32, out_f32=False, sm_count=152)
    assert cake_backend.k1_plan(K=65, **kw) == (64, 2)
    assert cake_backend.k1_plan(K=64, **kw) is None
    assert cake_backend.k1_splits_fit(1, 1) and not cake_backend.k1_splits_fit(0, 64)


def test_k1_rule_table_entries_decide_the_row(monkeypatch):
    kw = dict(L=1, M=4096, N=32, K=16231, out_f32=True)
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
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=212, **kw) is None
    monkeypatch.setitem(cake_backend.K1_MIN_SM_COUNT, "sm_107a", 0)
    assert cake_backend.k1_plan(arch="sm_107a", sm_count=212, **kw) == (64, 2)


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


# ---------------------------------------------------------------------------
# K2 planner (CPU)
# ---------------------------------------------------------------------------


def _segments(m_pairs, n_tiles, k_iters, units):
    """Each tile's segments ``(k_first, k_end)`` in K blocks, in K order, tile ``pair * n_tiles + column``: span
    ``s`` runs the (row-tile pair, K block) steps ``[s * steps / spans, (s + 1) * steps / spans)``, pair-major."""
    steps, spans = m_pairs * k_iters, units // n_tiles
    per_tile = [[] for _ in range(m_pairs * n_tiles)]
    for span in range(spans):
        first, last = span * steps // spans, (span + 1) * steps // spans
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
        # weight gradient: hidden 4096 and 6144 run the deterministic plan
        for hidden in (4096, 6144):
            assert (
                cake_backend.router_units(
                    "nn_t", arch=arch, sm_count=sms, M=hidden, N=256, K=16231
                )
                is None
            )
    # weight gradient, hidden 5120: the "waves" partition (whole waves of the CTA pairs of the device) on sm_100a /
    # sm_103a; sm_107a measured, missed (the partition would be 432 units)
    for arch in ("sm_100a", "sm_103a"):
        assert (
            cake_backend.router_units("nn_t", arch=arch, sm_count=152, **wgrad5120)
            == 380
        )
    assert cake_backend.ROUTER_RULES[("sm_107a", "nn_t", 5120, 256)] is None
    assert (
        cake_backend.router_units("nn_t", arch="sm_107a", sm_count=216, **wgrad5120)
        is None
    )
    assert (
        cake_backend.router_units("nn_t", arch="sm_100a", sm_count=148, **wgrad5120)
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
    # verdicts: per (arch, layout, hidden, experts); a forward verdict must keep two partials at most
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_100a", "nn_t", 6144, 256), "pairs"
    )
    assert (
        cake_backend.router_units("nn_t", arch="sm_100a", sm_count=148, **wgrad) == 74
    )
    monkeypatch.setitem(
        cake_backend.ROUTER_RULES, ("sm_103a", "kk", 6144, 256), "forward"
    )
    # one unit per CTA pair, at most one per tile (128 tiles here): every tile at most two partials
    assert cake_backend.router_units("kk", arch="sm_103a", sm_count=152, **fwd) == 76
    monkeypatch.setitem(cake_backend.ROUTER_RULES, ("sm_103a", "kk", 6144, 256), "S3")
    with pytest.raises(ValueError, match="more than two partials"):
        cake_backend.router_units("kk", arch="sm_103a", sm_count=152, **fwd)


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
    # hidden 5120: the "waves" partition (40 tiles, 129 K blocks: 3 waves of the 76 pairs)
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
    """Every template this checkout registers is one the host plans -- the transposed-store K1 programs
    (``dense_proj_gemm_atomic_nn_n<64|128|256>[_f32]_t``) and the two router classes -- or the weight gradient's
    head/tail program, registered before the planner names it."""
    wanted = {
        cake_backend.atomic_k1_template(bn, f32)
        for bn in cake_backend.K1_BLOCK_N_CHOICES
        for f32 in (False, True)
    } | {
        "router_fp32_gemm_atomic_kk",
        "router_fp32_gemm_atomic_nn_t",
        "router_fp32_gemm_atomic_nn_t_ht_c3_nreg",
    }
    for table in cake_jit.KERNELS.values():
        assert set(table) <= wanted, sorted(set(table) - wanted)


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
    cake_backend._ATOMIC_ARENAS[(99, 99)] = cake_backend._AtomicArena(
        torch.device("cpu"), (99, 99)
    )
    try:
        cake_backend.clear_convenience_caches()
        assert (
            key not in det_backend._K2_CACHE
            and key not in cake_backend._K1_ATOMIC_CACHE
        )
        assert (99, 99) not in cake_backend._ATOMIC_ARENAS
    finally:
        det_backend._K2_CACHE.pop(key, None)
        cake_backend._K1_ATOMIC_CACHE.pop(key, None)
        cake_backend._ATOMIC_ARENAS.pop((99, 99), None)


def test_the_split_k_workspace_kinds():
    """The K1 tickets and the K1 slots are zeroed when allocated or grown (the last split of a partial tile reads
    slot columns no split writes); the kinds are not the deterministic package's."""
    kinds = cake_backend._ATOMIC_ARENA_KINDS
    assert kinds["k1_tickets"] == (torch.uint32, True)
    assert kinds["k1_slots"] == (torch.float32, True)
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
    """``k1_rule(G, X, out, splits)``: a verdict for the row (and no SM-count floor), so the convenience path runs
    this package's program at that split count; returns the template."""

    def force(G, X, out, splits, block_n=None):
        arch = det_backend.arch_for(out.device)
        N, K = int(G.shape[1]), int(X.shape[1])
        block_n = block_n or cake_backend.k1_block_n(N)
        f32 = out.dtype == torch.float32
        _require_template(arch, cake_backend.atomic_k1_template(block_n, f32))
        key = (arch, 1, K, N, "f32" if f32 else "bf16")
        monkeypatch.setitem(cake_backend.K1_MIN_SM_COUNT, arch, 0)
        monkeypatch.delitem(cake_backend.K1_RULE_MIN_SM_COUNT, key, raising=False)
        monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, key, (block_n, splits))
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
        cake_backend.clear_convenience_caches()

    yield force
    cake_backend.clear_convenience_caches()


def _router_prepare(A, B, out, splits, arch):
    _require_template(
        arch,
        "router_fp32_gemm_atomic_" + ("nn_t" if A.stride(0) == 1 else "kk"),
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
    per_tile = _segments(plan.m_pairs, plan.n_tiles, plan.k_iters, plan.units)
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
    device, arch = _atomic_device_or_skip()
    X, W, G = _router_inputs(ROUTER_T, device, SEED + 20)
    A, B, out = _router_case(op, X, W, G)
    if spec is not None:
        router_rule(op, A, B, spec)
    prepared = _router_prepare(A, B, out, splits, arch)
    plan = prepared.plan
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
    for (rule_arch, L, M, N, dt), _ in sorted(
        cake_backend.K1_SPLIT_RULES.items(), key=lambda kv: kv[0]
    ):
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
    for (rule_arch, layout, hidden, _experts), spec in sorted(
        cake_backend.ROUTER_RULES.items(), key=lambda kv: kv[0]
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
        assert prepared.plan.units == plan.units
        _assert_ran(prepared.launch, prepared.module_name)
        torch.cuda.synchronize()
        assert _rel_fro(out, A.double() @ B.double()) <= ROUTER_REL_FRO_MAX
        ran += 1
    print(
        f"PLANNED_ROWS {arch} sm_count={sm_count}: {ran} rows ran this package's programs"
    )
    if not ran:
        pytest.skip(f"no verdict of {arch} applies on {sm_count} SMs")


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
