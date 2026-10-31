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


def _register(monkeypatch, arch: str, *templates: str) -> None:
    """Pretend this checkout registers ``templates`` on ``arch`` (CPU planner tests: nothing is loaded)."""
    table = dict(cake_jit.KERNELS.get(arch, {}))
    for template in templates:
        table[template] = "cake_dense_projection_gemm_" + "0" * 20
    monkeypatch.setitem(cake_jit.KERNELS, arch, table)


def test_k1_planner_plans_the_split_k_weight_gradient(monkeypatch):
    """The thin weight gradient ``out[32, 4096] = G.T @ X`` in its swapped orientation: one 256-row pair tile per
    256 input features, a 64-column tile, ``p`` filling the CTA pairs; the descriptor views are ``[L, K, M]`` /
    ``[L, K, N]``; an explicit split count (the control ``p = 1`` included) is honoured; an unregistered template
    keeps the row on the deterministic program."""
    arch, T, N, K = "sm_103a", 16231, 32, 4096
    G, X = (
        torch.zeros(T, N, dtype=torch.bfloat16),
        torch.zeros(T, K, dtype=torch.bfloat16),
    )
    out = torch.zeros(N, K, dtype=torch.bfloat16)
    kw = dict(transposed_out=True, arch=arch, sm_count=152)
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    plan, *_, reason = cake_backend.plan_atomic_gemm(X.t(), G, out, **kw)
    assert plan is None and "not registered" in reason
    _register(monkeypatch, arch, "dense_proj_gemm_atomic_nn_n64_t")
    plan, a_desc, b_desc, out3, reason = cake_backend.plan_atomic_gemm(
        X.t(), G, out, **kw
    )
    assert isinstance(plan, cake_backend.AtomicGemmPlan) and reason == ""
    assert (plan.L, plan.M, plan.N, plan.K, plan.block_n) == (1, K, N, T, 64)
    assert (plan.m_tiles, plan.n_tiles, plan.splits) == (
        16,
        1,
        4,
    )  # 76 pairs // 16 tiles
    assert (
        plan.grid == (2 * 64, 1, 1)
        and plan.template == "dense_proj_gemm_atomic_nn_n64_t"
    )
    assert plan.ws_f32_elems == 16 * 256 * 64 and plan.counters_u32 == 16 * 8
    assert tuple(a_desc.shape) == (1, T, K) and a_desc.stride(-1) == 1
    assert tuple(b_desc.shape) == (1, T, N) and b_desc.stride(-1) == 1
    assert tuple(out3.shape) == (1, N, K)
    for p in (1, 3, 254):
        assert (
            cake_backend.plan_atomic_gemm(X.t(), G, out, splits=p, **kw)[0].splits == p
        )
    with pytest.raises(ValueError, match="splits must be in"):
        cake_backend.plan_atomic_gemm(X.t(), G, out, splits=255, **kw)
    monkeypatch.setitem(cake_backend.K1_SPLIT_RULES, (arch, False, True, 1, K, N), 1)
    plan, *_, reason = cake_backend.plan_atomic_gemm(X.t(), G, out, **kw)
    assert plan is None and "p = 1" in reason


def test_k1_planner_keeps_rows_it_cannot_serve_on_the_deterministic_program(
    monkeypatch,
):
    arch, T = "sm_100a", 1000
    _register(
        monkeypatch,
        arch,
        *(cake_backend.atomic_k1_template(bn, False, False) for bn in (64, 128, 256)),
    )
    kw = dict(transposed_out=False, arch=arch, sm_count=148)

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


def test_k2_planner_plans_the_reduce_add_router(monkeypatch):
    arch, T, K, N = "sm_100a", 4099, 6144, 256
    X, W, G = torch.zeros(T, K), torch.zeros(N, K), torch.zeros(T, N)
    fwd = (X, W.t(), torch.zeros(T, N))
    wgrad = (G.t(), X, torch.zeros(N, K))
    kw = dict(arch=arch, sm_count=152)
    monkeypatch.setitem(cake_jit.KERNELS, arch, {})  # whatever this checkout registers
    plan, reason = cake_backend.plan_atomic_router(*fwd, splits=2, **kw)
    assert plan is None and "not registered" in reason
    _register(
        monkeypatch, arch, "router_fp32_gemm_atomic_kk", "router_fp32_gemm_atomic_nn_t"
    )
    plan, reason = cake_backend.plan_atomic_router(*fwd, **kw)
    assert plan is None and "S = 1" in reason  # the forward's model keeps one split
    plan, _ = cake_backend.plan_atomic_router(*fwd, splits=2, **kw)
    assert (plan.layout, plan.template, plan.splits, plan.swapped) == (
        "kk",
        "router_fp32_gemm_atomic_kk",
        2,
        False,
    )
    assert (plan.m_tiles, plan.n_tiles, plan.k_blocks, plan.k_iters_split) == (
        34,
        2,
        192,
        96,
    )
    assert plan.grid == (2 * 2 * 17 * 2, 1, 1)
    assert cake_backend.plan_atomic_router(*fwd, splits=1, **kw)[0].splits == 1
    with pytest.raises(ValueError, match="at most 2"):
        cake_backend.plan_atomic_router(*fwd, splits=3, **kw)
    plan, _ = cake_backend.plan_atomic_router(*wgrad, **kw)
    assert (plan.layout, plan.template, plan.swapped, plan.transposed_out) == (
        "nn_t",
        "router_fp32_gemm_atomic_nn_t",
        True,
        True,
    )
    assert (plan.M, plan.N, plan.K) == (K, N, T) and plan.splits > 1
    dgrad = (G, W, torch.zeros(T, K))
    assert cake_backend.plan_atomic_router(*dgrad, **kw) == (
        None,
        "the input gradient (kn) has one split",
    )
    # this package's program stores through an ``out`` 4 bytes past a 16-byte boundary
    shifted = torch.zeros(T * N + 1)[1:].view(T, N)
    assert (
        cake_backend.plan_atomic_router(X, W.t(), shifted, splits=2, **kw)[0]
        is not None
    )


def test_router_split_model_is_the_wave_model():
    """The provisional split model (until the measured per-architecture table): the forward keeps one split; the
    weight gradient minimises waves x (K blocks per unit + overhead)."""
    kw = dict(arch="sm_100a", N=256, K=16231, sm_count=152)
    got = [
        cake_backend._router_split_model("nn_t", M=H, **kw) for H in (4096, 5120, 6144)
    ]
    assert got == [7, 7, 3]
    assert (
        cake_backend._router_split_model(
            "kk", M=16231, N=256, K=6144, arch="sm_100a", sm_count=152
        )
        == 1
    )


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


# ---------------------------------------------------------------------------
# GPU: the non-deterministic programs.  Every test binds to the programs this checkout registers for the device
# (the planner's choice, or an explicit split count) and skips a row whose template is not registered.
# ---------------------------------------------------------------------------

# K1 weight-gradient rows ``out[N, K] = G[T, N].T @ X[T, K]``: label -> (N, K); the thin rows of the GLM-5.2 table
# (indexer heads, the 8192 -> 128 row, the kv_a projection) and a wide row.
K1_ROWS = {
    "indexer_hw": (32, 6144),
    "thin_48": (48, 5120),
    "indexer_k": (128, 6144),
    "kv_a": (576, 6144),
    "wide_512": (512, 4096),
}
K1_RAGGED_T = (16231, 129, 4097, 16172, 777)
ROUTER_T = 4099


def _k1_wgrad_inputs(row, T, out_dtype, device, seed):
    N, K = K1_ROWS[row]
    g = torch.Generator(device=device).manual_seed(seed)
    G, X = _bf16((T, N), device, g), _bf16((T, K), device, g)
    out = torch.full((N, K), float("nan"), device=device, dtype=out_dtype)
    return G, X, out


def _k1_wgrad_args(G, X, out):
    """The deterministic package's orientation of the weight gradient: ``(A, B, transposed_out)``."""
    return det_backend.wgrad_views(
        G, X, arch=det_backend.arch_for(out.device), out_f32=out.dtype == torch.float32
    )


def _k1_plan_or_skip(A, B, out, transposed, **knobs):
    facts = det_backend.device_facts(out.device)
    plan = cake_backend.plan_dense_projection_gemm(
        A,
        B,
        out,
        transposed_out=transposed,
        sm_count=facts.sm_count,
        l2_bytes=facts.l2_bytes,
        deterministic=False,
        **knobs,
    )
    if not isinstance(plan, cake_backend.AtomicGemmPlan):
        pytest.skip(
            f"this row runs the deterministic program on {det_backend.arch_for(out.device)}"
        )
    return plan


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


def _arenas_zero():
    for arena in cake_backend._ATOMIC_ARENAS.values():
        for kind in ("k1_acc", "k1_tickets"):
            buf = arena.buffers.get(kind)
            if (
                buf is not None
                and buf.to(torch.int64 if kind == "k1_tickets" else torch.float32).any()
            ):
                return False
    return True


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("row", sorted(K1_ROWS))
def test_gpu_k1_non_deterministic_budgets_over_reruns(row, out_dtype):
    """The split-K weight gradient (planner's split count) meets the deterministic budgets against FP64 on every
    one of ``RERUNS`` prepared launches and convenience calls, each into a NaN-poisoned ``out``; its distance to
    the deterministic result stays within the same budget; the workspace is zero after every launch."""
    device = _det_device_or_skip()
    G, X, out = _k1_wgrad_inputs(row, 16231, out_dtype, device, SEED)
    A, B, transposed = _k1_wgrad_args(G, X, out)
    _k1_plan_or_skip(A, B, out, transposed)
    ref = G.double().t() @ X.double()
    det_out = cake_backend.projection_wgrad(G, X, torch.empty_like(out))
    prepared = cake_backend.prepare_projection_wgrad(G, X, out, deterministic=False)
    assert isinstance(prepared, cake_backend.PreparedAtomicGemm) and prepared.splits > 1
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, ref, f"prepared rerun {i}")
        _k1_check(out, det_out.double(), f"vs det rerun {i}")
        assert prepared.workspace_is_zero(), i
        out.fill_(float("nan"))
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
        torch.cuda.synchronize()
        _k1_check(out, ref, f"convenience rerun {i}")
    assert _arenas_zero()


@pytest.mark.parametrize("splits", [1, 2, 3, 8])
def test_gpu_k1_explicit_split_counts(splits):
    """Every explicit split count (``1``: this package's program storing whole tiles, the control of the
    reduction) gives the budgeted result and leaves the workspace zero."""
    device = _det_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 1)
    A, B, transposed = _k1_wgrad_args(G, X, out)
    _k1_plan_or_skip(A, B, out, transposed, splits=splits)
    prepared = cake_backend.prepare_dense_projection_gemm(
        A, B, out, transposed_out=transposed, deterministic=False, splits=splits
    )
    assert (
        isinstance(prepared, cake_backend.PreparedAtomicGemm)
        and prepared.splits == splits
    )
    prepared.launch()
    torch.cuda.synchronize()
    _k1_check(out, G.double().t() @ X.double())
    assert prepared.workspace_is_zero()


def test_gpu_k1_batched_mla_head_views():
    """The batched MLA weight gradient (64 heads, strided head views of ``[T, H, slot]`` activations): this
    package's split-K program at an explicit split count (one workspace slab per head tile), then the generic
    entry point with the planner's choice."""
    device = _det_device_or_skip()
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
    _k1_plan_or_skip(A, B, out, False, splits=2)
    ref = A.double() @ B.double()
    prepared = cake_backend.prepare_dense_projection_gemm(
        A, B, out, deterministic=False, splits=2
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicGemm)
    assert (prepared.plan.L, prepared.splits) == (H, 2)
    for _ in range(3):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        _k1_check(out, ref)
        assert prepared.workspace_is_zero()
        out.fill_(float("nan"))
        cake_backend.dense_projection_gemm(A, B, out, deterministic=False)
        torch.cuda.synchronize()
        _k1_check(out, ref)
    assert _arenas_zero()


def test_gpu_k1_ragged_token_counts_reuse_the_poisoned_workspace():
    """Changing ``T`` up and down on one stream: the convenience path reuses its arena (grown, never shrunk) and
    every output -- poisoned with NaN before each call -- meets the budget; the zero invariant holds throughout."""
    device = _det_device_or_skip()
    cake_backend.clear_convenience_caches()
    for i, T in enumerate(K1_RAGGED_T):
        G, X, out = _k1_wgrad_inputs(
            "thin_48", T, torch.bfloat16, device, SEED + 10 + i
        )
        if i == 0:
            A, B, transposed = _k1_wgrad_args(G, X, out)
            _k1_plan_or_skip(A, B, out, transposed)
        cake_backend.projection_wgrad(G, X, out, deterministic=False)
        torch.cuda.synchronize()
        _k1_check(out, G.double().t() @ X.double(), f"T={T}")
        assert _arenas_zero(), T
    assert len(cake_backend._ATOMIC_ARENAS) == 1
    (arena,) = cake_backend._ATOMIC_ARENAS.values()
    assert arena.version >= 2  # created, then grown or reused; never shrunk


def _router_inputs(T, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    K, N = 6144, 256
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


def _router_prepare_or_skip(A, B, out, splits):
    plan = cake_backend.plan_router_fp32_gemm(
        A, B, out, splits=splits, deterministic=False
    )
    if not isinstance(plan, cake_backend.AtomicRouterPlan):
        pytest.skip(
            f"the router layout runs the deterministic program on {det_backend.arch_for(out.device)}"
        )
    prepared = cake_backend.prepare_router_fp32_gemm(
        A, B, out, splits=splits, deterministic=False
    )
    assert isinstance(prepared, cake_backend.PreparedAtomicRouterGemm)
    return prepared


@pytest.mark.parametrize(
    "op, splits",
    [
        ("fwd", 1),
        ("fwd", 2),
        ("wgrad", 1),
        ("wgrad", None),
        ("wgrad", 4),
        ("wgrad", 11),
    ],
)
def test_gpu_router_d11_gate(op, splits):
    """The router's accuracy gate, the deterministic package's shipped budget: ``rel_fro <= 2e-6`` against FP64 of
    the exact fp32 operands, finite, worst over ``RERUNS`` launches into a NaN-poisoned ``out``; the forward's top-8
    agrees with FP64 on every row and repeats bit for bit.  Against the deterministic program at the same split
    count: one split and two splits give its bits (``0 + a + b == 0 + b + a``); more splits stay within the
    reorder bound ``2 (S - 1) 2^-24 sum_k |A| |B|``."""
    device = _det_device_or_skip()
    X, W, G = _router_inputs(ROUTER_T, device, SEED + 20)
    A, B, out = _router_case(op, X, W, G)
    prepared = _router_prepare_or_skip(A, B, out, splits)
    S = prepared.splits
    ref = A.double() @ B.double()
    det_out = torch.empty_like(out)
    cake_backend.router_fp32_gemm(
        A, B, det_out, splits=S
    )  # the deterministic program at the same split count
    bound = 2 * (S - 1) * 2.0**-24 * (A.double().abs() @ B.double().abs())
    first, worst = None, 0.0
    for i in range(RERUNS):
        out.fill_(float("nan"))
        prepared.launch()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out).all()), i
        worst = max(worst, _rel_fro(out, ref))
        if S <= 2:
            assert torch.equal(out, det_out), (i, S)
        else:
            assert bool(((out.double() - det_out.double()).abs() <= bound).all()), (
                i,
                S,
            )
        if op == "fwd":
            if first is None:
                first = out.clone()
                ref_top = torch.topk(ref, 8, dim=1).indices.sort(dim=1).values
                top = torch.topk(out.double(), 8, dim=1).indices.sort(dim=1).values
                assert int((top != ref_top).any(dim=1).sum()) == 0
            assert torch.equal(out, first), i
    assert worst <= ROUTER_REL_FRO_MAX, worst


def test_gpu_router_entry_points_reach_the_reduce_add_program():
    """``router_forward`` / ``router_wgrad`` with ``deterministic=False``: the forward stays reproducible and its
    top-8 is the deterministic forward's; both meet the gate."""
    device = _det_device_or_skip()
    X, W, G = _router_inputs(ROUTER_T, device, SEED + 21)
    _router_prepare_or_skip(
        G.t(), X, torch.empty(G.shape[1], X.shape[1], device=device), None
    )
    first = cake_backend.router_forward(X, W, deterministic=False, splits=2)
    det_fwd = cake_backend.router_forward(X, W)
    assert torch.equal(
        first.topk(8, dim=1).indices.sort(dim=1).values,
        det_fwd.topk(8, dim=1).indices.sort(dim=1).values,
    )
    ref_fwd, ref_wgrad = X.double() @ W.double().t(), G.double().t() @ X.double()
    for _ in range(RERUNS):
        out = cake_backend.router_forward(X, W, deterministic=False, splits=2)
        assert torch.equal(out, first) and _rel_fro(out, ref_fwd) <= ROUTER_REL_FRO_MAX
        wgrad = cake_backend.router_wgrad(G, X, deterministic=False)
        assert _rel_fro(wgrad, ref_wgrad) <= ROUTER_REL_FRO_MAX


def _profile_kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]


def _det_kernel(name: str) -> bool:
    """A kernel of the deterministic package: its programs' symbols and its template bodies' namespace."""
    return "dense_projection_det_gemm" in name


def test_gpu_non_deterministic_mode_launches_this_packages_programs():
    """Kernel-name evidence: ``deterministic=False`` launches this package's ``kernel_cake_dense_projection_gemm_*``
    programs (none of the deterministic package's K1 / K2 programs for the reduced operation), the default only
    the deterministic package's."""
    device = _det_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 30)
    A, B, transposed = _k1_wgrad_args(G, X, out)
    plan = _k1_plan_or_skip(A, B, out, transposed)
    ours = cake_jit.MODULES[
        cake_jit.select_module(det_backend.arch_for(device), plan.template)
    ]["kernel"]
    nondet = _profile_kernel_names(
        lambda: cake_backend.projection_wgrad(G, X, out, deterministic=False)
    )
    assert any(ours in name for name in nondet), nondet
    assert not any(_det_kernel(name) for name in nondet), nondet
    det = _profile_kernel_names(lambda: cake_backend.projection_wgrad(G, X, out))
    assert any(_det_kernel(name) for name in det), det
    assert not any("kernel_cake_dense_projection_gemm_" in name for name in det), det
    Xf, Wf, Gf = _router_inputs(1001, device, SEED + 31)
    A2, B2, out2 = _router_case("wgrad", Xf, Wf, Gf)
    prepared = _router_prepare_or_skip(A2, B2, out2, 4)
    names = _profile_kernel_names(prepared.launch)
    assert any(
        cake_jit.MODULES[prepared.module_name]["kernel"] in name for name in names
    ), names


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


def test_gpu_det_then_non_det_share_a_stream_not_a_workspace():
    """A deterministic call and a non-deterministic call of one geometry on one stream: each package keeps its own
    arena (the deterministic package's uninitialised slabs never reach this package's zero-invariant workspace)."""
    device = _det_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 50)
    A, B, transposed = _k1_wgrad_args(G, X, out)
    _k1_plan_or_skip(A, B, out, transposed)
    cake_backend.clear_convenience_caches()
    ref = G.double().t() @ X.double()
    for _ in range(3):
        det = cake_backend.projection_wgrad(G, X, torch.empty_like(out))
        nondet = cake_backend.projection_wgrad(G, X, out, deterministic=False)
        torch.cuda.synchronize()
        _k1_check(det, ref)
        _k1_check(nondet, ref)
    ours = {
        id(b) for a in cake_backend._ATOMIC_ARENAS.values() for b in a.buffers.values()
    }
    theirs = {id(b) for a in det_backend._ARENAS.values() for b in a.buffers.values()}
    assert ours and theirs and not ours & theirs
    assert _arenas_zero()


def test_gpu_concurrent_streams_keep_their_own_workspaces():
    """Two streams issuing non-deterministic K1 and K2 calls of different geometries concurrently: one arena per
    (device, stream), every result within budget, the zero invariant kept."""
    device = _det_device_or_skip()
    cake_backend.clear_convenience_caches()
    streams = [torch.cuda.Stream(device) for _ in range(2)]
    cases = []
    for i, (row, T) in enumerate((("indexer_k", 16231), ("thin_48", 4097))):
        G, X, out = _k1_wgrad_inputs(row, T, torch.bfloat16, device, SEED + 60 + i)
        if i == 0:
            A, B, transposed = _k1_wgrad_args(G, X, out)
            _k1_plan_or_skip(A, B, out, transposed)
        cases.append((G, X, out))
    Xf, Wf, Gf = _router_inputs(ROUTER_T, device, SEED + 62)
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
    assert _arenas_zero()


def test_gpu_prepared_launches_allocate_nothing_and_capture_into_a_graph():
    """The prepared K1 and K2 launches allocate nothing on the device, capture into one CUDA graph (K2: the split,
    the memset of ``out`` and the kernel) and replay with fresh operand values copied in place: the replay equals
    the eager launch's budgeted result, and the K1 workspace is zero after the replays."""
    device = _det_device_or_skip()
    G, X, out = _k1_wgrad_inputs("indexer_k", 16231, torch.bfloat16, device, SEED + 70)
    A, B, transposed = _k1_wgrad_args(G, X, out)
    _k1_plan_or_skip(A, B, out, transposed)
    k1 = cake_backend.prepare_projection_wgrad(G, X, out, deterministic=False)
    Xf, Wf, Gf = _router_inputs(ROUTER_T, device, SEED + 71)
    A2, B2, out2 = _router_case("wgrad", Xf, Wf, Gf)
    k2 = _router_prepare_or_skip(A2, B2, out2, 4)
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
        assert k1.workspace_is_zero()
