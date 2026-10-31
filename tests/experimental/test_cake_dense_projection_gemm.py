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

# The dense projection GEMM package with an explicit numerical mode (flashinfer-ai/flashinfer#8069): the dispatch of
# every entry point (``deterministic=True`` -> the deterministic package's canonical object, ``deterministic=False``
# -> the routes of ``cake_backend.ROUTES``), PyTorch's deterministic-algorithms rule, the names of the former import
# path, the registry and file names of this package's programs and its zero-invariant workspace kinds.  The CPU tests
# launch nothing.  The GPU tests (``test_gpu_*``) need a device the deterministic package registers programs for;
# the non-deterministic ones also need a program of this package on that device and skip otherwise.

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
NON_DET_TOKEN = re.compile(r"non_?det", re.IGNORECASE)


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
    if not cake_jit.__name__.startswith("flashinfer."):
        pytest.skip("a vendored copy has no sibling package")
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
        assert kwargs == {"transposed_out": False, "knobs": {}}
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
    (launch_args, launch_kwargs), (prepare_args, _) = atomic.calls
    A, B, transposed = det_backend.wgrad_views(G, X, arch=None, out_f32=False)
    assert launch_args[0] == "launch" and prepare_args[0] == "prepare"
    assert launch_args[1].equal(A) and launch_args[2].equal(B) and launch_args[3] is out
    assert launch_kwargs == {
        "transposed_out": transposed,
        "knobs": {"sk": "auto", "cta_rows": None},
    }


def test_k2_routes(monkeypatch):
    """``deterministic=False``: the router forward and weight gradient reach this package's reduce-add path; the
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


def test_the_split_paths_need_registered_programs():
    """While this checkout registers no program of the non-deterministic schedule, its paths refuse instead of
    running anything else."""
    if cake_jit.MODULES:
        pytest.skip("this checkout registers non-deterministic programs")
    A, B, out = _k1_views("nn")
    with pytest.raises(NotImplementedError, match="not registered"):
        cake_backend.dense_projection_gemm(A, B, out, deterministic=False)
    X, W = torch.zeros(64, 32), torch.zeros(16, 32)
    with pytest.raises(NotImplementedError, match="not registered"):
        cake_backend.router_forward(X, W, deterministic=False)


# ---------------------------------------------------------------------------
# PyTorch's deterministic-algorithms mode (CPU)
# ---------------------------------------------------------------------------

# fmt: off
NON_DET_CALLS = {
    "dense_projection_gemm": lambda: cake_backend.dense_projection_gemm(*_k1_views("nn"), deterministic=False),
    "projection_forward": lambda: cake_backend.projection_forward("X", "W", deterministic=False),
    "projection_wgrad": lambda: cake_backend.projection_wgrad("G", "X", "out", deterministic=False),
    "router_forward": lambda: cake_backend.router_forward("X", "W", deterministic=False),
    "router_dgrad": lambda: cake_backend.router_dgrad("G", "W", deterministic=False),
    "router_wgrad": lambda: cake_backend.router_wgrad("G", "X", deterministic=False),
    "split_fp32_to_bf16x3_": lambda: cake_backend.split_fp32_to_bf16x3_("s", "p", "r", deterministic=False),
}
# fmt: on


@pytest.mark.parametrize("name", sorted(NON_DET_CALLS))
def test_explicit_false_under_torch_deterministic_mode_raises(
    monkeypatch, torch_determinism, name
):
    recorder = _patch_det(monkeypatch, name)
    monkeypatch.setattr(cake_backend, "_atomic_k1", Recorder())
    monkeypatch.setattr(cake_backend, "_atomic_k2", Recorder())
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        NON_DET_CALLS[name]()
    assert not recorder.calls


@pytest.mark.parametrize("name", sorted(NON_DET_CALLS))
def test_explicit_false_under_warn_only_warns_and_runs_the_deterministic_path(
    monkeypatch, torch_determinism, name
):
    recorder = _patch_det(monkeypatch, name)
    atomic1, atomic2 = Recorder(), Recorder()
    monkeypatch.setattr(cake_backend, "_atomic_k1", atomic1)
    monkeypatch.setattr(cake_backend, "_atomic_k2", atomic2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match="running the deterministic path"):
        assert NON_DET_CALLS[name]() is recorder.result
    assert len(recorder.calls) == 1 and not atomic1.calls and not atomic2.calls


# ---------------------------------------------------------------------------
# Planner hooks (CPU)
# ---------------------------------------------------------------------------


def test_router_splits_policy(monkeypatch):
    kw = dict(arch="sm_100a", M=16231, N=256, K=6144, sm_count=152)
    assert cake_backend.router_splits("kn", **kw) == 1
    assert cake_backend.router_splits("kn", splits=4, **kw) == 1
    assert cake_backend.router_splits("kk", splits=2, **kw) == 2
    assert cake_backend.router_splits("nn_t", splits=11, **kw) == 11
    with pytest.raises(ValueError, match="at most 2"):
        cake_backend.router_splits("kk", splits=3, **kw)
    with pytest.raises(ValueError, match=">= 1"):
        cake_backend.router_splits("nn_t", splits=0, **kw)
    monkeypatch.setitem(cake_backend.ROUTER_SPLIT_RULES, ("sm_100a", "nn_t"), 7)
    assert cake_backend.router_splits("nn_t", **kw) == 7


def test_k1_split_parts_reads_the_rules(monkeypatch):
    kw = dict(
        arch="sm_103a",
        L=1,
        M=4096,
        N=128,
        K=16231,
        out_f32=False,
        transposed_out=True,
        sm_count=152,
    )
    monkeypatch.setitem(
        cake_backend.K1_SPLIT_RULES, ("sm_103a", False, True, 1, 4096, 128), 5
    )
    assert cake_backend.k1_split_parts(**kw) == 5


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


def test_no_det_token_in_generated_names_and_no_non_det_token_anywhere():
    root = Path(cake_jit.__file__).parent
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and "aot" not in p.parts
    ]
    for path in files:
        rel = path.relative_to(root).as_posix()
        assert not DET_TOKEN.search(path.name), rel
        assert not NON_DET_TOKEN.search(path.read_text(errors="replace")), rel
    generated = [*cake_jit.MODULES, *(r["kernel"] for r in cake_jit.MODULES.values())]
    generated += [t for table in cake_jit.KERNELS.values() for t in table]
    assert not [n for n in generated if DET_TOKEN.search(n)]
    manifest = root / "build.json"
    if manifest.exists():
        document = json.loads(manifest.read_text())
        assert document["package"] == package.__name__
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
    key = ("fi-8069-test", object())
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


def test_the_split_k_workspace_kinds_are_zeroed_and_separate():
    kinds = cake_backend._ATOMIC_ARENA_KINDS
    assert kinds["k1_acc"] == (torch.float32, True) and kinds["k1_tickets"] == (
        torch.uint32,
        True,
    )
    assert not set(kinds) & {
        "k1_ws",
        "k1_counters",
        "k2_partials",
    }  # the deterministic package's kinds
    arena = cake_backend._AtomicArena(torch.device("cpu"), (0, 0))
    acc = arena.require("k1_acc", 100)
    assert acc.dtype == torch.float32 and acc.numel() == 100 and not acc.any()
    assert arena.require("k1_acc", 50) is acc and arena.version == 1
    grown = arena.require("k1_acc", 1000)
    assert grown.numel() == 1000 and not grown.any() and arena.version == 2
    tickets = arena.require("k1_tickets", 3)
    assert (
        tickets.dtype == torch.uint32
        and tickets.numel() == 16
        and not tickets.to(torch.int64).any()
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
    device = _det_device_or_skip()
    arch = det_backend.arch_for(device)
    if not cake_jit.KERNELS.get(arch):
        pytest.skip(f"this checkout registers no non-deterministic program for {arch}")
    return device, arch


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


def test_gpu_router_non_deterministic_accuracy_and_reproducible_forward():
    """The router's forward (at most two splits) is bitwise reproducible over reruns and keeps the top-k of the
    deterministic forward; forward and weight gradient meet the FP32-class budget against FP64 on every rerun."""
    device, _ = _atomic_device_or_skip()
    g = torch.Generator(device=device).manual_seed(SEED)
    T, K, N = 4099, 6144, 256
    X = torch.randn(T, K, device=device, generator=g)
    W = torch.randn(N, K, device=device, generator=g) * 0.02
    G = torch.randn(T, N, device=device, generator=g)
    ref_fwd = X.double() @ W.double().t()
    ref_wgrad = G.double().t() @ X.double()
    first = cake_backend.router_forward(X, W, deterministic=False)
    det_fwd = cake_backend.router_forward(X, W)
    assert torch.equal(first.topk(8, dim=1).indices, det_fwd.topk(8, dim=1).indices)
    worst_fwd = worst_wgrad = 0.0
    for _ in range(RERUNS):
        out = cake_backend.router_forward(X, W, deterministic=False)
        assert torch.equal(out, first)
        worst_fwd = max(worst_fwd, _rel_fro(out, ref_fwd))
        worst_wgrad = max(
            worst_wgrad,
            _rel_fro(cake_backend.router_wgrad(G, X, deterministic=False), ref_wgrad),
        )
    assert worst_fwd <= ROUTER_REL_FRO_MAX and worst_wgrad <= ROUTER_REL_FRO_MAX


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [16231, 4097, 129])
def test_gpu_weight_gradient_non_deterministic_accuracy(T, out_dtype):
    """The split-K weight gradient meets the deterministic budgets against FP64 on every one of the reruns, with a
    poisoned reused output, and leaves its workspace zero."""
    device, _ = _atomic_device_or_skip()
    g = torch.Generator(device=device).manual_seed(SEED + T)
    K, N = 4096, 128
    G, X = _bf16((T, N), device, g), _bf16((T, K), device, g)
    ref = G.double().t() @ X.double()
    out = torch.full((N, K), float("nan"), device=device, dtype=out_dtype)
    for _ in range(RERUNS):
        out.fill_(float("nan"))
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
        if out_dtype == torch.bfloat16:
            bad = (out.double() - ref).abs() > BF16_ATOL + BF16_RTOL * ref.abs()
            assert not bad.any()
        else:
            assert _rel_fro(out, ref) <= F32_REL_FRO_MAX
            assert float(
                (out.double() - ref).abs().max()
            ) <= F32_MAX_ABS_OVER_REF_ABSMAX * float(ref.abs().max())
    torch.cuda.synchronize()
    for arena in cake_backend._ATOMIC_ARENAS.values():
        for kind in ("k1_acc", "k1_tickets"):
            if kind in arena.buffers:
                assert (
                    not arena.buffers[kind]
                    .to(torch.int64 if kind == "k1_tickets" else torch.float32)
                    .any()
                )
