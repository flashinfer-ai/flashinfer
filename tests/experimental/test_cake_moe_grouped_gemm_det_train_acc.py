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

Weight-gradient accumulation of the deterministic ragged BF16 MoE grouped GEMM
package: ``cake_moe_grouped_gemm_det_train_prepare_wgrad`` / ``cake_moe_grouped_gemm_det_train_wgrad`` with
``accumulate=True`` (``out[e] += G[e].T @ X[e]``) and ``cake_moe_grouped_gemm_det`` with
``fuse_wgrad_accumulation=True`` (``w.main_grad += dW`` in the backward).  The package
adds on one of two routes (``plan["accumulate"]``): ``"fused"``, the accumulate builds
(``<stage>_acc``) of the 512-wide k tile's main stage and, for a split tail, of the
ordered tail reduce (``tail_reduce_acc``) add into ``out`` in place; ``"composed"``, the
overwrite of the same plan into a scratch of ``out``'s dtype owned by the launch, then
``out.add_`` on the same stream.  The oracle of every GPU case is AccumulateGrad of the
deterministic ``dW``: ``out0.clone().add_(dW)`` with ``dW`` the overwrite of the same call
in ``out``'s dtype (an fp32 ``main_grad`` gets one fp32 add of the fp32 ``dW``; a bf16 one
the bf16 ``dW`` added in bf16), bitwise on both routes, and bitwise run to run; the fused
route's ``out`` is bitwise the composed route's (int32 view).

On the CPU (no program is loaded): the registry's accumulate builds against the exact
set this release ships (``SHIPPED_ACC_STAGES``, which the README feature matrix states
too) and against their twins, the ``build.json`` entries, the signatures, the accumulate
plans against the overwrite plans (the owner rows on every architecture at the SM counts
of its parts), the route choice (the 256-wide k tile, a stage without an accumulate
build, a split tail whose route serves no ``tail_reduce_acc``, a build that cannot load
by the loader's static rule, the alignment), the output the programs are bound to
(``out`` or the scratch), the forced-form shapes the GPU cases use (split and unsplit
tails on every architecture), the checks of ``out`` on both routes, ``sum_m == 0``, the
launch's add or version bump (once per launch, after the programs, outside autograd,
inference tensors, no alert under torch's deterministic mode), the facade's handoff and
the autograd flag's contract with CPU stand-ins for the preparations (the Megatron
placeholder, ``zero_out_wgrad``, ``retain_graph``, ``torch.autograd.grad`` without
``w``, a frozen ``w``, ``create_graph``, reentrant, non-reentrant and nested
checkpoints, a re-pointed ``main_grad``, a direct ``apply``, trace and compile).  On a
GPU: every weight-gradient form of the device with an unsplit and a split tail, fp32 and
bf16 ``out``, against the oracle, run to run and against the fp64 reference; the fused
route against the composed route bitwise (split tails, the planned walk with empty
groups at its end, the wave halves, the k256 tail, ``-0.0`` and subnormals in ``out0``,
up to 32 groups, ``FUSED_REPEATS`` launches); empty groups, ``sum_m == 0``, row padding,
one group, the ``m_indptr`` form, an inference tensor, two streams, CUDA graph replays,
allocations, the version counter on both routes, torch's deterministic mode, the owner
rows, and the autograd flag with the generated programs.  A fused case the shipped set
promises fails unless the call is fused; it skips (``"cannot load"`` in the reason) only
where the build's library cannot load in this environment (neither prebuilt nor a JIT
toolkit at its ``min_cuda``), where the call composes instead.
"""

import collections
import contextlib
import importlib.util
import inspect
import os
import shutil
import sys
import types
import warnings

import pytest
import torch
import torch.utils.checkpoint

from tests.experimental import test_cake_moe_grouped_gemm_det_train as base

pkg = base.pkg
cb = base.cb
cake_jit = base.cake_jit
cake_autograd = base.cake_autograd
HOST_PLAN_CONSTANTS = base.HOST_PLAN_CONSTANTS
SUPPORTED_COMPUTE_CAPABILITIES = base.SUPPORTED_COMPUTE_CAPABILITIES
ROUTES = base.ROUTES
PROGRAMS = base.PROGRAMS
prepare_wgrad = cb.cake_moe_grouped_gemm_det_train_prepare_wgrad

DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}
ACC = cb.ACC_SUFFIX
TAIL_REDUCE = cb.TAIL_REDUCE
TAIL_REDUCE_ACC = cb.TAIL_REDUCE_ACC
# The main stages with an accumulate build: the planner-reachable 512-wide k tile forms.
TWIN_FORMS = ("main_halves", "main_wave_halves", "main_halves_k256t")
F32_K256 = cake_jit.record_name("wgrad", out_dtype="float32", tile_k=256)
F32_K512 = cake_jit.record_name("wgrad", out_dtype="float32", tile_k=512)
# The accumulate builds this release ships, route -> architecture -> stages: fp32 ``out``
# only (a bf16 ``out`` composes: the bf16 overwrite and a same-dtype ``add_``).  The fp32
# 512-wide k tile route serves its twins' builds and the tail reduce's; the fp32 256-wide
# k tile route serves the tail reduce's (the split tail of ``main_halves_k256t``; one
# library, built for every architecture).  The registry serves exactly these
# (``test_acc_bindings_are_the_shipped_set``) and the README feature matrix states the
# same (``test_docs_list_the_shipped_acc_builds``).
SHIPPED_ACC_STAGES = {
    F32_K512: {
        "sm_100a": ("main_halves_acc", "main_wave_halves_acc", TAIL_REDUCE_ACC),
        "sm_103a": ("main_halves_acc", "main_wave_halves_acc", TAIL_REDUCE_ACC),
        "sm_107a": ("main_halves_acc", "main_halves_k256t_acc", TAIL_REDUCE_ACC),
    },
    F32_K256: {
        "sm_100a": (TAIL_REDUCE_ACC,),
        "sm_103a": (TAIL_REDUCE_ACC,),
        "sm_107a": (TAIL_REDUCE_ACC,),
    },
}
# The output dtypes with accumulate builds (the fused cases run these).
FUSED_DTYPES = tuple(
    dtype
    for dtype, name in DTYPE_NAMES.items()
    if cake_jit.record_name("wgrad", out_dtype=name, tile_k=512) in SHIPPED_ACC_STAGES
)
# Launches of each route per fused-against-composed GPU case (run to run).
FUSED_REPEATS = 5
# Empty groups where the 512-wide k tile's order puts them last (the planned walk drops
# the trailing empty tiles, which must still add exact zeros), and a split tail beside
# empty groups: the kernel tests' lists.
EMPTY_TAIL_SIZES = (
    [640, 384, 768, 448, 512, 576, 0, 0],
    [4099] * 4 + [0, 0] + [130] * 2,
)
# 32 groups: empty, one-row, < 64-row and non-tile-aligned groups beside long ones.
E32_SIZES = (
    [0, 1, 63, 65, 0, 127, 129, 255]
    + [257, 0, 511, 513]
    + [1031] * 4
    + [
        2049,
        0,
        4097,
        33,
    ]
    + [3001] * 12
)
INT_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
FP32_SUBNORMAL = 1e-40
# The SM counts of the parts of each architecture (B200 / B300: 148, GB300: 160,
# R200: 208 / 212 / 216); the CPU plan checks run at each.
SM_COUNTS = {"sm_100a": (148, 160), "sm_103a": (148, 160), "sm_107a": (208, 212, 216)}
ARCH_SM_COUNTS = [(arch, sms) for arch, counts in SM_COUNTS.items() for sms in counts]
# The weight-gradient forms: label -> (k tile, main stage).  ``k256`` / ``k256_wave`` are the
# 256-wide k tile's plain and wave-synchronised stages; the others the 512-wide k tile's.
FORMS = {
    "k256": (256, "main"),
    "k256_wave": (256, "main_wave"),
    "main": (512, "main"),
    "main_wave": (512, "main_wave"),
    "main_halves": (512, "main_halves"),
    "main_wave_halves": (512, "main_wave_halves"),
    "main_halves_k256t": (512, "main_halves_k256t"),
}
# The group sizes a forced-form case tries, in order (an empty and a one-row group each).
CASE_SIZES = (
    [4100, 0, 4170, 1],
    [1200, 0, 1, 2800],
    [2400, 0, 1, 5600],
    [4800, 0, 1, 11200],
    [9600, 0, 1, 22400],
)
# Launches of one accumulate per GPU case, each from a fresh copy of ``out0``: every
# result must carry the bits of the first (run to run).
REPEATS = 10


def _balanced(sum_m, groups):
    """``groups`` sizes summing to ``sum_m``, the last taking the remainder."""
    size = sum_m // groups
    return [size] * (groups - 1) + [sum_m - size * (groups - 1)]


# The owner rows (#5678 / the package benchmark): (label, sizes) over both projections.
OWNER_ROWS = tuple(
    (label, _balanced(sum_m, groups)) for label, sum_m, groups, _ in base.BENCH_ROWS
)


class _NoModule:
    """A stand-in for a loaded program module: every entry is a no-op."""

    def __getattr__(self, name):
        return lambda *args: None


def _fake_device(monkeypatch, arch, sms, *, checks=True, loadable=True):
    """Run the host of ``cake_moe_grouped_gemm_det_train_prepare_wgrad`` without a GPU: the operands'
    device stands for an ``arch`` device with ``sms`` SMs, programs load as no-op modules,
    the descriptor slots are plain bytes and every accumulate build's library counts as
    loadable (``loadable``; the loader's rule has its own tests).  ``checks=False`` also
    skips the accumulate checks of ``out`` and its alignment (for meta tensors, which have
    no addresses; the checks have their own tests)."""
    monkeypatch.setattr(
        cb, "_check_device", lambda tensors: next(iter(tensors.values())).device
    )
    monkeypatch.setattr(cb, "arch_for", lambda device: arch)
    monkeypatch.setattr(cb, "sm_count", lambda device: sms)
    monkeypatch.setattr(cb, "load_module", lambda *args, **kwargs: _NoModule())
    monkeypatch.setattr(
        cb,
        "_aligned_bytes",
        lambda n, device, what: torch.empty(n, dtype=torch.uint8, device=device),
    )
    monkeypatch.setattr(cb, "_device_index", lambda device: 0)
    monkeypatch.setattr(cb, "_WARNED", set())
    monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: loadable)
    if not checks:
        monkeypatch.setattr(cb, "_check_accumulate_out", lambda *args: None)
        monkeypatch.setattr(cb, "_accumulate_aligned", lambda out: True)


def _acc_bindings():
    """``(route, stage, arch)`` of every accumulate stage the registry serves."""
    found = set()
    for name, route in ROUTES.items():
        for item in route["stages"]:
            if not item["name"].endswith(ACC):
                continue
            for arch in route["arches"]:
                if arch in PROGRAMS[item["program"]]["arches"]:
                    found.add((name, item["name"], arch))
    return sorted(found)


def _shipped_bindings():
    """``(route, stage, arch)`` of every accumulate build of ``SHIPPED_ACC_STAGES``."""
    return sorted(
        (name, stage, arch)
        for name, arches in SHIPPED_ACC_STAGES.items()
        for arch, stages in arches.items()
        for stage in stages
    )


def _promised(arch, form, out_dtype):
    """Whether this release ships the accumulate build of the 512-wide k tile's ``form`` for
    ``out_dtype`` on ``arch`` (and so the tail reduce's, which every such route ships)."""
    name = cake_jit.record_name("wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=512)
    return form + ACC in SHIPPED_ACC_STAGES.get(name, {}).get(arch, ())


def _tail_route(plan, out_dtype):
    """The route whose tail-reduce stage sums the split tail of ``plan`` (the 256-wide k tile's
    for ``main_halves_k256t``), None when the tail is not split."""
    if plan["tail_splits"] <= 1:
        return None
    return cake_jit.record_name(
        "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=plan["tail_tile_k"]
    )


def _prepare_pair(g, x, offs, out, num_groups=None):
    """The overwrite and the accumulate preparation of the same call (no launch)."""
    ovw = prepare_wgrad(g, x, offs, out=out, num_groups=num_groups)
    acc = prepare_wgrad(g, x, offs, out=out, num_groups=num_groups, accumulate=True)
    return ovw, acc


def _assert_twin_launch(ovw, acc, arch):
    """``acc`` is ``ovw`` in accumulate mode: the same route and plan apart from the
    accumulate keys, the route ``wgrad_accumulate_mode`` gives (a reason exactly when
    composed); fused: the overwrite's stages with ``_acc`` each, no scratch, the version
    bump; composed: the overwrite's stages and a contiguous scratch of ``out``'s shape and
    dtype.  Neither zero-fills."""
    assert ovw.plan["accumulate"] == "off" and "accumulate_reason" not in ovw.plan
    assert acc.record_name == ovw.record_name
    assert _plan_without(acc.plan, "accumulate", "accumulate_reason") == _plan_without(
        ovw.plan, "accumulate"
    )
    mode = acc.plan["accumulate"]
    assert mode in cb.ACCUMULATE_MODES and mode != "off"
    assert ("accumulate_reason" in acc.plan) == (mode == "composed"), acc.plan
    assert not acc.zero_fill
    if not ovw.stages:  # no row: nothing launches either way
        assert acc.stages == () and acc.add_from is None and not acc.bump_version
        return mode
    want = cb.wgrad_accumulate_mode(
        ovw.record_name,
        ovw.stages[0],
        ovw.plan["tile_k"],
        arch,
        reduce_name=_tail_route(ovw.plan, acc.out.dtype),
        aligned=cb._accumulate_aligned(acc.out),
    )
    assert (mode, acc.plan.get("accumulate_reason")) == want, (ovw.stages, arch)
    assert len(acc.launches) == len(ovw.launches)
    if mode == "fused":
        assert acc.stages == tuple(stage + ACC for stage in ovw.stages)
        assert acc.add_from is None and acc.bump_version
    else:
        assert acc.stages == ovw.stages and not acc.bump_version
        scratch = acc.add_from
        assert scratch is not None and scratch is not acc.out
        assert scratch.shape == acc.out.shape and scratch.dtype == acc.out.dtype
        assert scratch.is_contiguous()
    return mode


def _record_binds(monkeypatch):
    """Record ``(stage, kwargs)`` of every ``_bind`` (the bound programs' arguments)."""
    bound = []
    real = cb._bind

    def record(name, stage, kwargs, arch, **extra):
        bound.append((stage, kwargs))
        return real(name, stage, kwargs, arch, **extra)

    monkeypatch.setattr(cb, "_bind", record)
    return bound


def _operands(sizes, n, k, out_dtype, device):
    sum_m, e = sum(sizes), len(sizes)
    g = torch.zeros(sum_m, n, dtype=torch.bfloat16, device=device)
    x = torch.zeros(sum_m, k, dtype=torch.bfloat16, device=device)
    if device == torch.device("meta"):
        offs = torch.empty(e, dtype=torch.int32, device=device)
    else:
        offs = torch.tensor(sizes, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    out = torch.zeros(e, n, k, dtype=out_dtype, device=device)
    return g, x, offs, out


def _plan_without(plan, *keys):
    return {key: value for key, value in plan.items() if key not in keys}


def _force_form(monkeypatch, form, arch):
    """Constants under which a weight gradient on ``arch`` takes ``form`` (``FORMS``; given a
    ``K`` of ``_form_ks(form)``; the 512-wide k tile at any rows per group); False where the
    architecture has no such form."""
    c = HOST_PLAN_CONSTANTS
    tile, stage = FORMS[form]
    halves = stage in ("main_halves", "main_wave_halves", "main_halves_k256t")
    wave = stage in ("main_wave", "main_wave_halves")
    if halves and arch not in c["k512_release_archs"]:
        return False
    if wave and arch not in c["wave_sync_archs"][tile]:
        return False
    if stage == "main_halves_k256t" and arch not in c["k256_tail_archs"]:
        return False
    if tile == 256:
        monkeypatch.setitem(c, "k512_archs", [])
    elif arch not in c["k512_archs"]:
        return False
    else:
        monkeypatch.setitem(c["k512_min_rows_per_group"], arch, 1)
    if stage == "main_halves_k256t":
        monkeypatch.setitem(c, "k256_tail_min_rows", 1)
    else:
        monkeypatch.setitem(c, "k256_tail_archs", [])
    if not halves:
        monkeypatch.setitem(c, "k512_release_archs", [])
    if not wave:
        monkeypatch.setitem(c, "wave_sync_min_k", 1 << 30)
    return True


def _form_ks(form):
    """Reductions ``K`` a shape of ``form`` may take."""
    c = HOST_PLAN_CONSTANTS
    tile, stage = FORMS[form]
    if stage in ("main_wave", "main_wave_halves"):
        return [int(c["wave_sync_min_k"]) + tile * i for i in range(4)]
    if stage == "main_halves_k256t":
        return [k for k in (2048, 1536, 1024) if k <= int(c["k256_tail_max_k"])]
    if tile == 256:
        return [512, 768, 1024, 1536, 2048]
    return [1024, 1536, 2048, 3072]


def _det_plan(sizes, n, k, arch, sms, out_dtype=torch.float32):
    """``(main stage, plan)`` that ``cake_moe_grouped_gemm_det_train_prepare_wgrad`` takes for ``sizes`` and
    ``(N, K)`` on ``arch`` with ``sms`` SMs, under the constants as the test set them."""
    c = HOST_PLAN_CONSTANTS
    sum_m, e = sum(sizes), len(sizes)
    tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
    name = cake_jit.record_name(
        "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=tile_k
    )
    stage, _wave = cb.select_wgrad_stage(
        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=e
    )
    plan = cb.plan_wgrad(
        sum_m=sum_m,
        num_groups=e,
        n=n,
        k=k,
        tile_k=tile_k,
        sm_count=sms,
        reduce_threads=256,
        k256_tail=stage == "main_halves_k256t",
        constants=c,
    )
    return stage, plan


def _find_case(arch, sms, form, split, out_dtype=torch.float32):
    """``(sizes, n, k)``: the first group sizes of ``CASE_SIZES``, ``K`` of ``_form_ks`` and
    ``N`` up to 48 column blocks whose plan on ``arch`` with ``sms`` SMs takes ``form`` and
    splits its tail (``split``: ``tail_splits > 1``) or not, within the 32-bit element
    addressing; None when no candidate does.  The constants must force ``form`` already."""
    tile, stage = FORMS[form]
    span = int(HOST_PLAN_CONSTANTS["int32_elems"])
    for sizes in CASE_SIZES:
        for k in _form_ks(form):
            for blocks in range(1, 49):
                n = 256 * blocks
                if len(sizes) * (n + 2) * (k + 32) > span:
                    break
                got, plan = _det_plan(sizes, n, k, arch, sms, out_dtype)
                if got != stage or plan["tile_k"] != tile:
                    break  # the stage does not depend on N
                if (plan["tail_splits"] > 1) == split:
                    return list(sizes), n, k
    return None


# ---------------------------------------------------------------------------
# The host (CPU)
# ---------------------------------------------------------------------------


def test_accumulate_keywords_and_signatures():
    """The new switches are keyword-only and off by default (also the accumulate query of
    ``generated_program_available``); the autograd function's ``forward`` takes
    ``fuse_wgrad_accumulation`` after the internal deferred-launch slot; no deterministic
    entry grows a ``deterministic`` parameter; the accumulate modes are ``"off"``,
    ``"fused"`` and ``"composed"``."""
    for fn, keyword in (
        (cb.cake_moe_grouped_gemm_det_train_prepare_wgrad, "accumulate"),
        (pkg.cake_moe_grouped_gemm_det_train_wgrad, "accumulate"),
        (cb.generated_program_available, "accumulate"),
        (pkg.cake_moe_grouped_gemm_det, "fuse_wgrad_accumulation"),
        (cake_autograd.cake_moe_grouped_gemm_det, "fuse_wgrad_accumulation"),
    ):
        parameters = inspect.signature(fn).parameters
        assert parameters[keyword].kind is inspect.Parameter.KEYWORD_ONLY, fn
        assert parameters[keyword].default is False, fn
        assert "deterministic" not in parameters, fn
    forward = inspect.signature(
        cake_autograd.CakeMoEGroupedGemmDetFunction.forward
    ).parameters
    assert list(forward)[-2:] == ["_launch", "fuse_wgrad_accumulation"]
    assert forward["fuse_wgrad_accumulation"].default is False
    assert cb.ACCUMULATE_MODES == ("off", "fused", "composed")


@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_accumulate_plans_equal_the_overwrite_plans(arch, sm_count, monkeypatch):
    """On every owner row and projection, for fp32 and bf16 ``out``: the accumulate
    preparation takes the overwrite's route, launch count and plan (apart from
    ``plan["accumulate"]`` / ``plan["accumulate_reason"]``).  An fp32 ``out`` is fused
    wherever the shipped set has the main stage's build (every owner row takes a
    half-release stage, so every fp32 owner row on every architecture): the overwrite's
    stages with ``_acc`` (the ordered tail reduce's build too when the tail splits), every
    program bound to ``out`` itself.  A bf16 ``out`` composes (``"stage"``): the
    overwrite's stages bound to a contiguous scratch of ``out``'s shape and dtype owned by
    the launch.  Neither zero-fills; the overwrite plans of the two dtypes are equal."""
    _fake_device(monkeypatch, arch, sm_count, checks=False)
    bound = _record_binds(monkeypatch)
    meta = torch.device("meta")
    for label, sizes in OWNER_ROWS:
        for proj, n, k in base.BENCH_PROJ:
            plans = {}
            for out_dtype in (torch.float32, torch.bfloat16):
                what = (label, proj, out_dtype)
                g, x, offs, out = _operands(sizes, n, k, out_dtype, meta)
                bound.clear()
                ovw = prepare_wgrad(g, x, offs, out=out)
                ovw_bound = list(bound)
                bound.clear()
                acc = prepare_wgrad(g, x, offs, out=out, accumulate=True)
                mode = _assert_twin_launch(ovw, acc, arch)
                assert ovw.stages and "halves" in ovw.stages[0], what
                assert not ovw.zero_fill and ovw.add_from is None, what
                assert acc.out is out, what
                if _promised(arch, ovw.stages[0], out_dtype):
                    assert mode == "fused", (what, acc.plan)
                    target, ldc, stride_e = out, k, n * k
                else:
                    assert (mode, acc.plan["accumulate_reason"]) == (
                        "composed",
                        "stage",
                    ), what
                    target, ldc, stride_e = acc.add_from, k, n * k
                assert [s for s, _ in bound] == list(acc.stages), what
                for (_s, kwargs), (_t, ovw_kwargs) in zip(
                    bound, ovw_bound, strict=True
                ):
                    assert kwargs["C"] is target and ovw_kwargs["C"] is out, what
                    assert kwargs["ldc"] == ldc and kwargs["stride_e"] == stride_e, what
                plans[out_dtype] = ovw.plan
            assert plans[torch.float32] == plans[torch.bfloat16], (label, proj)


def test_row_padded_out_binds_a_contiguous_scratch(monkeypatch):
    """A row- and expert-padded ``out`` is accepted; the programs write the contiguous
    scratch (row stride ``K``, expert stride ``N * K``), not the padded span, so the
    padding is never read or written."""
    _fake_device(monkeypatch, "sm_100a", 148)
    bound = _record_binds(monkeypatch)
    sizes, n, k = [64, 0, 64], 256, 256
    g, x, offs, _ = _operands(sizes, n, k, torch.float32, torch.device("cpu"))
    buffer = torch.zeros(len(sizes), n + 2, k + 32)
    out = buffer[:, :n, :k]
    acc = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert acc.out is out and acc.add_from.is_contiguous()
    for _stage, kwargs in bound:
        assert kwargs["C"] is acc.add_from
        assert (kwargs["ldc"], kwargs["stride_e"]) == (k, n * k)
    bound.clear()
    prepare_wgrad(g, x, offs, out=out)
    for _stage, kwargs in bound:
        assert (kwargs["ldc"], kwargs["stride_e"]) == (k + 32, (n + 2) * (k + 32))


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_forced_form_cases_exist(arch, sm_count, out_dtype, monkeypatch):
    """Every weight-gradient form the architecture serves for ``out_dtype`` has a shape
    of ``CASE_SIZES`` with a split tail (the ordered tail reduce runs: ``tail_splits >
    1``) and one without, at every SM count of its parts, so the GPU cases of each form
    and dtype run both (none passes vacuously); on sm_107a the k256-tail twin's split tail
    is the 256-wide k tile's."""
    found = []
    for form, (tile, stage) in FORMS.items():
        with monkeypatch.context() as mp:
            if not _force_form(mp, form, arch):
                continue
            name = cake_jit.record_name(
                "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=tile
            )
            if not cake_jit.stage_served(name, stage, arch):
                continue
            for split in (True, False):
                case = _find_case(arch, sm_count, form, split, out_dtype)
                assert case is not None, (form, split)
                sizes, n, k = case
                _stage, plan = _det_plan(sizes, n, k, arch, sm_count, out_dtype)
                if split and stage == "main_halves_k256t":
                    assert plan["tail_tile_k"] == cb.K256_TAIL_TILE_K
            found.append(form)
    assert "main_halves" in found and "k256" in found, found
    if arch == "sm_107a":
        assert "main_halves_k256t" in found, found


# ---------------------------------------------------------------------------
# The accumulate builds and the route choice (CPU)
# ---------------------------------------------------------------------------


def test_acc_bindings_are_the_shipped_set():
    """The registry serves exactly the accumulate builds of ``SHIPPED_ACC_STAGES``: the
    fp32 512-wide k tile route serves ``main_halves_acc`` on sm_100a / sm_103a / sm_107a,
    ``main_wave_halves_acc`` on sm_100a / sm_103a, ``main_halves_k256t_acc`` on sm_107a and
    ``tail_reduce_acc`` on all three; the fp32 256-wide k tile route serves
    ``tail_reduce_acc`` (the k256 tail's split tail); the bf16 routes serve none (a bf16
    ``out`` composes).  A render that lost or added an accumulate stage fails here."""
    assert _acc_bindings() == _shipped_bindings()
    for arch in cake_jit.ARCHES:
        for name in DTYPE_NAMES.values():
            route = cake_jit.record_name("wgrad", out_dtype=name, tile_k=512)
            want = any(
                stage != TAIL_REDUCE_ACC
                for stage in SHIPPED_ACC_STAGES.get(route, {}).get(arch, ())
            )
            assert cb.acc_stage_served(name, arch) == want, (arch, name)
    assert list(FUSED_DTYPES) == [torch.float32]


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _doc_matrix_row(path, label):
    """The first line of ``path`` whose table row starts with ``label``."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip("| ").startswith(label):
                return line.rstrip("\n")
    raise AssertionError(f"{path} has no row {label!r}")


def test_docs_list_the_shipped_acc_builds():
    """The README feature-matrix row ``wgrad accumulate, fused`` names, per architecture,
    the stages of ``SHIPPED_ACC_STAGES`` on the fp32 512-wide k tile route (the twins and
    the tail reduce) for an fp32 ``out`` only (no bf16)."""
    readme = os.path.join(
        _REPO,
        "flashinfer",
        "experimental",
        "cake",
        "cake_moe_grouped_gemm_det_train",
        "README.md",
    )
    row = _doc_matrix_row(readme, "`wgrad` accumulate, fused")
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    label, per_arch = cells[0], cells[1:]
    assert "fp32" in label and "bf16" not in label, label
    assert len(per_arch) == len(cake_jit.ARCHES), row
    for arch, cell in zip(cake_jit.ARCHES, per_arch, strict=True):
        forms = {
            name + ACC for name in cell.replace("`", "").replace(" ", "").split(",")
        }
        assert forms == set(SHIPPED_ACC_STAGES[F32_K512][arch]), (arch, cell)


def test_acc_stages_are_builds_of_their_twins():
    """Every accumulate stage the registry serves is ``<twin>_acc`` on a weight-gradient
    route, for a twin the route serves on the same architecture: a planner-reachable
    half-release form on the 512-wide k tile route, or the tail reduce; its program is
    another build of the family with the twin program's argument plan and launch shape
    (so the overwrite's bindings and reduce grid serve it), under the twin's
    specializations.  Every route with a main stage's build serves ``tail_reduce_acc`` on
    the architecture (a split tail can fuse), and ``main_halves_k256t_acc`` is served only
    where ``main_halves_acc`` is and the 256-wide k tile route serves ``tail_reduce_acc``."""
    bindings = _acc_bindings()
    assert bindings, "the registry serves no accumulate build"
    for name, stage, arch in bindings:
        route = ROUTES[name]
        assert route["op"] == "wgrad", name
        twin = stage[: -len(ACC)]
        assert twin in TWIN_FORMS + (TAIL_REDUCE,), (name, stage)
        if twin != TAIL_REDUCE:
            assert name.endswith("_k512"), (name, stage)
            assert cake_jit.stage_served(name, TAIL_REDUCE_ACC, arch), (name, arch)
        assert cake_jit.stage_served(name, twin, arch), (name, stage, arch)
        acc_binding = cake_jit.stage_binding(name, stage, arch)
        twin_binding = cake_jit.stage_binding(name, twin, arch)
        assert acc_binding["specializations"] == twin_binding["specializations"]
        acc_program = PROGRAMS[acc_binding["program"]]
        twin_program = PROGRAMS[twin_binding["program"]]
        assert acc_binding["program"] != twin_binding["program"], (name, stage)
        assert acc_program["library"] != twin_program["library"]
        assert acc_program["arg_plan"] == twin_program["arg_plan"], (name, stage)
        assert acc_program["launch"] == twin_program["launch"], (name, stage)
        for label in (acc_binding["program"], acc_program["library"]):
            assert label.startswith("cake_moe_grouped_det_gemm"), label
        if twin == "main_halves_k256t":
            assert cake_jit.stage_served(name, "main_halves" + ACC, arch), (name, arch)
            assert cake_jit.stage_served(F32_K256, TAIL_REDUCE_ACC, arch), arch
        dtype = "float32" if "_f32_" in name else "bfloat16"
        if twin != TAIL_REDUCE:
            assert cb.acc_stage_served(dtype, arch)


def test_overwrite_routes_never_bind_an_acc_stage():
    """With ``accumulate=False`` the stage rules never select an accumulate stage."""
    c = HOST_PLAN_CONSTANTS
    for name, route in ROUTES.items():
        if route["op"] != "wgrad":
            continue
        tile_k = 512 if name.endswith("_k512") else 256
        for arch in route["arches"]:
            for k in (1024, 2048, 6144):
                for sum_m, groups in ((131072, 4), (131072, 32), (4096, 4)):
                    stage, _wave = cb.select_wgrad_stage(
                        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=groups
                    )
                    assert not stage.endswith(ACC), (name, arch, k, stage)


@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_k256_tile_and_forced_forms_compose(arch, sm_count, monkeypatch):
    """Every weight-gradient form the constants can force on the architecture, with and
    without a split tail: the 256-wide k tile composes (``"tile_k"``), and so do the
    512-wide k tile's forms without an accumulate build (``main``, ``main_wave``:
    ``"stage"``) and every form into a bf16 ``out`` (``"stage"``), each with the overwrite
    plan and stages; the half-release forms into an fp32 ``out`` are fused."""
    _fake_device(monkeypatch, arch, sm_count, checks=False)
    meta = torch.device("meta")
    seen = []
    for form, (tile, stage) in FORMS.items():
        with monkeypatch.context() as mp:
            if not _force_form(mp, form, arch):
                continue
            for out_dtype in (torch.float32, torch.bfloat16):
                name = cake_jit.record_name(
                    "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=tile
                )
                if not cake_jit.stage_served(name, stage, arch):
                    continue
                for split in (False, True):
                    case = _find_case(arch, sm_count, form, split, out_dtype)
                    assert case is not None, (form, split)
                    g, x, offs, out = _operands(*case, out_dtype, meta)
                    ovw, acc = _prepare_pair(g, x, offs, out)
                    assert ovw.stages[0] == stage, (form, ovw.stages)
                    mode = _assert_twin_launch(ovw, acc, arch)
                    if ovw.plan["tile_k"] == 256:
                        want = ("composed", "tile_k")
                    elif _promised(arch, stage, out_dtype):
                        want = ("fused", None)
                    else:
                        want = ("composed", "stage")
                    assert (mode, acc.plan.get("accumulate_reason")) == want, (
                        form,
                        out_dtype,
                        split,
                    )
                    seen.append(want[1])
    assert {"tile_k", "stage"} <= set(seen), seen


def _record_bound_routes(monkeypatch):
    """Record ``(route, stage, kwargs)`` of every ``_bind``."""
    bound = []
    real = cb._bind

    def record(name, stage, kwargs, arch, **extra):
        bound.append((name, stage, kwargs))
        return real(name, stage, kwargs, arch, **extra)

    monkeypatch.setattr(cb, "_bind", record)
    return bound


@pytest.mark.parametrize("arch,sm_count", ARCH_SM_COUNTS)
def test_split_tail_fuses_with_the_tail_reduce_acc(arch, sm_count, monkeypatch):
    """The deterministic split tail is ordered: its partials are summed in a fixed order
    by the tail reduce, whose accumulate build adds that sum into C once.  For every
    shipped half-release form on the architecture, a forced shape with a split tail
    (``tail_splits > 1``) into an fp32 ``out`` is fused with the stages ``(<form>_acc,
    tail_reduce_acc)``, the tail reduce bound on the route of the tail (the 256-wide k
    tile's for ``main_halves_k256t``) and both programs bound to ``out`` in place; an
    unsplit shape is fused with ``<form>_acc`` alone.  With the tail route's
    ``tail_reduce_acc`` hidden, the split-tail call composes (``"tail"``) with the
    overwrite's stages while the unsplit one stays fused."""
    _fake_device(monkeypatch, arch, sm_count, checks=False)
    bound = _record_bound_routes(monkeypatch)
    meta = torch.device("meta")
    f32 = torch.float32
    forms = [form for form in TWIN_FORMS if _promised(arch, form, f32)]
    assert forms, arch
    real_served = cb.stage_served
    for form in forms:
        with monkeypatch.context() as mp:
            assert _force_form(mp, form, arch), form
            for split in (True, False):
                case = _find_case(arch, sm_count, form, split, f32)
                assert case is not None, (form, split)
                g, x, offs, out = _operands(*case, f32, meta)
                bound.clear()
                acc = prepare_wgrad(g, x, offs, out=out, accumulate=True)
                assert acc.plan["accumulate"] == "fused", (form, split, acc.plan)
                assert (acc.plan["tail_splits"] > 1) == split
                if split:
                    tail = cake_jit.record_name(
                        "wgrad", out_dtype="float32", tile_k=acc.plan["tail_tile_k"]
                    )
                    assert tail == (
                        F32_K256 if form == "main_halves_k256t" else F32_K512
                    )
                    assert acc.stages == (form + ACC, TAIL_REDUCE_ACC)
                    assert [(n, st) for n, st, _ in bound] == [
                        (F32_K512, form + ACC),
                        (tail, TAIL_REDUCE_ACC),
                    ]
                else:
                    assert acc.stages == (form + ACC,)
                    assert [(n, st) for n, st, _ in bound] == [(F32_K512, form + ACC)]
                for _n, _st, kwargs in bound:
                    assert kwargs["C"] is out
                mp.setattr(
                    cb,
                    "stage_served",
                    lambda name, stage, a: (
                        stage != TAIL_REDUCE_ACC and real_served(name, stage, a)
                    ),
                )
                ovw, hidden = _prepare_pair(g, x, offs, out)
                _assert_twin_launch(ovw, hidden, arch)
                if split:
                    assert (
                        hidden.plan["accumulate"],
                        hidden.plan["accumulate_reason"],
                    ) == ("composed", "tail"), form
                    assert hidden.stages == ovw.stages
                else:
                    assert hidden.plan["accumulate"] == "fused", form
                mp.setattr(cb, "stage_served", real_served)


def test_misaligned_out_composes_with_one_warning(monkeypatch):
    """An fp32 ``out`` whose base, row pitch or expert pitch is not ``ACC_ALIGN``-byte
    aligned takes the composed route (``"alignment"``) with the overwrite plan, and the
    first such preparation in the process warns once (``UserWarning`` naming the caller);
    an aligned ``out`` of the same call is fused."""
    arch = "sm_100a"
    _fake_device(monkeypatch, arch, 148)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"], arch, 1)
    sizes, n, k = [64, 0, 64], 256, 512
    e = len(sizes)
    g, x, offs, aligned = _operands(sizes, n, k, torch.float32, torch.device("cpu"))
    ovw, acc = _prepare_pair(g, x, offs, aligned)
    _assert_twin_launch(ovw, acc, arch)
    assert _promised(arch, ovw.stages[0], torch.float32), ovw.stages
    assert acc.plan["accumulate"] == "fused" and "accumulate_reason" not in acc.plan
    assert cb.ACC_ALIGN % 4 == 0
    step = 1  # one fp32 element: misaligned for any ACC_ALIGN above 4 bytes
    flat = torch.zeros(e * (n + 1) * (k + 8) + 8, dtype=torch.float32)
    misaligned = {
        "base": flat[step : step + e * n * k].view(e, n, k),
        "row pitch": flat.as_strided((e, n, k), (n * (k + 1), k + 1, 1)),
        "expert pitch": flat.as_strided((e, n, k), (n * k + 1, k, 1)),
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for what, out in misaligned.items():
            assert not cb._accumulate_aligned(out), what
            ovw, acc = _prepare_pair(g, x, offs, out)
            _assert_twin_launch(ovw, acc, arch)
            assert (acc.plan["accumulate"], acc.plan["accumulate_reason"]) == (
                "composed",
                "alignment",
            ), what
    user = [w for w in rec if w.category is UserWarning]
    assert len(user) == 1, [str(w.message) for w in user]
    assert "aligned" in str(user[0].message) and "composed" in str(user[0].message)
    assert os.path.samefile(user[0].filename, __file__), user[0].filename


def test_unloadable_acc_build_composes_with_one_warning(monkeypatch):
    """Where an accumulate build's library cannot load by the loader's static rule (no
    prebuilt library and a JIT toolkit below its ``min_cuda``), the call composes
    (``"min_cuda"``) with the overwrite plan and binds only the overwrite programs, while
    loading an accumulate build would raise the loader's ``min_cuda`` error; the first
    such preparation warns once.  The main stage's build and the tail reduce's are checked
    alike: a split tail whose tail-reduce build alone cannot load composes too.
    ``generated_program_available(accumulate=True)`` follows the same rule."""
    arch = "sm_103a"
    sms = 160
    _fake_device(monkeypatch, arch, sms, loadable=False)
    acc_programs = {
        cake_jit.stage_binding(name, stage, a)["program"]
        for name, stage, a in _shipped_bindings()
        if a == arch
    }
    loaded = []

    def load(program, arch, specializations=None):
        if program in acc_programs:
            raise RuntimeError(
                f"{program} needs CUDA 13.4 or newer; the local toolkit is 13.0"
            )
        loaded.append(program)
        return _NoModule()

    monkeypatch.setattr(cb, "load_module", load)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"], arch, 1)
    sizes, n, k = [64, 0, 64], 256, 512
    g, x, offs, out = _operands(sizes, n, k, torch.float32, torch.device("cpu"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(2):
            ovw, acc = _prepare_pair(g, x, offs, out)
            _assert_twin_launch(ovw, acc, arch)
            assert _promised(arch, ovw.stages[0], torch.float32), ovw.stages
            assert (acc.plan["accumulate"], acc.plan["accumulate_reason"]) == (
                "composed",
                "min_cuda",
            )
    assert loaded and not set(loaded) & acc_programs
    user = [w for w in rec if w.category is UserWarning]
    assert len(user) == 1, [str(w.message) for w in user]
    assert "min_cuda" in str(user[0].message) and "composed" in str(user[0].message)
    assert os.path.samefile(user[0].filename, __file__), user[0].filename
    # Only the tail reduce's build cannot load: a split tail composes, an unsplit one fuses.
    monkeypatch.setattr(
        cb, "acc_library_loadable", lambda name, stage, a: stage != TAIL_REDUCE_ACC
    )
    with monkeypatch.context() as mp:
        assert _force_form(mp, "main_halves", arch)
        for split, want in ((True, ("composed", "min_cuda")), (False, ("fused", None))):
            case = _find_case(arch, sms, "main_halves", split)
            assert case is not None, split
            meta = _operands(*case, torch.float32, torch.device("meta"))
            mp.setattr(cb, "_check_accumulate_out", lambda *args: None)
            mp.setattr(cb, "_accumulate_aligned", lambda out: True)
            mp.setattr(cb, "load_module", lambda *args, **kwargs: _NoModule())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                acc = prepare_wgrad(*meta[:3], out=meta[3], accumulate=True)
            assert (acc.plan["accumulate"], acc.plan.get("accumulate_reason")) == want
    device = torch.device("cpu")
    monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: False)
    assert not cb.generated_program_available(
        device, "wgrad", torch.float32, accumulate=True
    )
    monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: True)
    assert cb.generated_program_available(
        device, "wgrad", torch.float32, accumulate=True
    )


def _never_loads(*args, **kwargs):
    raise AssertionError("acc_library_loadable must not build or load a library")


class _Path:
    def __init__(self, exists):
        self.exists = exists

    def is_file(self):
        return self.exists


def _cuda_version(text):
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def _library(name, stage, arch):
    binding = cake_jit.stage_binding(name, stage, arch)
    return cake_jit.library_name(binding["program"], arch, binding["specializations"])


@pytest.mark.parametrize("name,stage,arch", _shipped_bindings())
def test_acc_library_loadable_is_the_loader_rule(name, stage, arch, monkeypatch):
    """``acc_library_loadable`` is the loader's own rule, decided without building,
    loading or probing anything: a prebuilt library under ``aot/`` loads whatever the
    toolkit; otherwise the JIT toolkit must reach the ``build.json`` entry's
    ``min_cuda``; without FlashInfer's JIT nothing builds."""
    library = _library(name, stage, arch)
    floor = _cuda_version(cake_jit.cake_build.libraries_by_name()[library]["min_cuda"])
    toolkit_of_process = cb._jit_cuda_version
    for loader in ("load_library", "jit_spec"):
        monkeypatch.setattr(cake_jit.cake_build, loader, _never_loads)
    below = (floor[0], floor[1] - 1) if floor[1] else (floor[0] - 1, 9)
    for prebuilt, toolkit, want in (
        (True, None, True),
        (True, below, True),
        (False, None, False),
        (False, below, False),
        (False, floor, True),
        (False, (floor[0], floor[1] + 1), True),
    ):
        monkeypatch.setattr(
            cb,
            "prebuilt_path",
            lambda lib, prebuilt=prebuilt: _Path(prebuilt and lib == library),
        )
        monkeypatch.setattr(cb, "_jit_cuda_version", lambda toolkit=toolkit: toolkit)
        assert cb.acc_library_loadable(name, stage, arch) is want, (prebuilt, toolkit)
    with pytest.MonkeyPatch.context() as mp:
        # FlashInfer's JIT is not importable (a prebuilt-only or vendored copy): the real
        # ``cake_build._jit_toolchain`` path, whose error names a manifest library.
        mp.delenv("FLASHINFER_DISABLE_JIT", raising=False)
        mp.setitem(sys.modules, "flashinfer.jit.core", None)
        assert toolkit_of_process.__wrapped__() is None


def test_jit_toolkit_needs_nvcc_and_the_jit_enabled(tmp_path, monkeypatch):
    """FlashInfer's JIT toolkit counts only where it can build: with ``FLASHINFER_DISABLE_JIT``
    set, without ``bin/nvcc`` under FlashInfer's CUDA path (where its version query falls
    back to PyTorch's CUDA version), or when that path cannot be resolved (an ``OSError``),
    ``_jit_cuda_version`` is None and an accumulate build that is not prebuilt cannot load
    (the call composes, ``"min_cuda"``)."""
    cpp_ext = pytest.importorskip("flashinfer.jit.cpp_ext")
    toolkit = cb._jit_cuda_version.__wrapped__
    monkeypatch.setattr(
        cpp_ext, "get_cuda_version", lambda: types.SimpleNamespace(major=99, minor=0)
    )
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: str(tmp_path))
    monkeypatch.setattr(cb, "_jit_cuda_version", toolkit)
    monkeypatch.setattr(cb, "prebuilt_path", lambda lib: _Path(False))
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    shipped = _shipped_bindings()
    assert toolkit() is None  # no bin/nvcc
    assert not any(cb.acc_library_loadable(*binding) for binding in shipped)
    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("")
    assert toolkit() == (99, 0)
    assert all(cb.acc_library_loadable(*binding) for binding in shipped)
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    assert toolkit() is None
    assert not any(cb.acc_library_loadable(*binding) for binding in shipped)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT")

    def no_which():
        raise FileNotFoundError("which")

    monkeypatch.setattr(cpp_ext, "get_cuda_path", no_which)
    assert toolkit() is None
    assert not any(cb.acc_library_loadable(*binding) for binding in shipped)


def test_generated_program_available_with_accumulate(monkeypatch):
    """``generated_program_available(..., accumulate=True)`` is True exactly where the
    512-wide k tile route of the dtype serves a main stage's accumulate build on the
    architecture whose library can load (either dtype when ``out_dtype`` is None); False
    for a dtype the weight gradient does not write, and for every dtype where no
    accumulate build can load; it refuses a non-bool switch and another operation."""
    device = torch.device("cpu")
    for loadable in (True, False):
        monkeypatch.setattr(cb, "acc_library_loadable", lambda *args, ok=loadable: ok)
        for arch in cake_jit.ARCHES:
            monkeypatch.setattr(cb, "arch_for", lambda d, arch=arch: arch)
            served = {}
            for out_dtype, name in DTYPE_NAMES.items():
                route = cake_jit.record_name("wgrad", out_dtype=name, tile_k=512)
                served[out_dtype] = any(
                    stage != TAIL_REDUCE_ACC
                    for stage in SHIPPED_ACC_STAGES.get(route, {}).get(arch, ())
                )
                got = cb.generated_program_available(
                    device, "wgrad", out_dtype, accumulate=True
                )
                assert got == (loadable and served[out_dtype]), (arch, out_dtype)
            assert cb.generated_program_available(
                device, "wgrad", None, accumulate=True
            ) == (loadable and any(served.values()))
            for other in (torch.float16, torch.int32):
                assert not cb.generated_program_available(
                    device, "wgrad", other, accumulate=True
                ), (arch, other)
    with pytest.raises(ValueError, match="weight gradient"):
        cb.generated_program_available(device, "fwd", accumulate=True)
    with pytest.raises(ValueError, match="accumulate must be a bool"):
        cb.generated_program_available(device, "wgrad", accumulate=1)


def _load_vendored_build(tmp_path):
    """``cake_build`` of a copy of the package's loader and ``build.json`` under another name."""
    source = cake_jit.cake_build.__file__
    root = tmp_path / "vendored_acc_build"
    root.mkdir()
    shutil.copy(source, root / "cake_build.py")
    shutil.copy(cake_jit.cake_build._MANIFEST_PATH, root / "build.json")
    spec = importlib.util.spec_from_file_location(
        "vendored_det_acc_build_loader", root / "cake_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_manifest_lists_the_acc_libraries(tmp_path):
    """``build.json`` has an entry for every shipped accumulate build (program,
    architecture, specializations), whose ``min_cuda`` is at least its twin's; the main
    stages' fp32 builds (the PTX ISA 9.4 ``red.global.add.noftz.f32`` reduction) need
    CUDA 13.4.  A vendored copy of the loader and the manifest lists them too."""
    libraries = cake_jit.cake_build.libraries_by_name()
    vendored = _load_vendored_build(tmp_path).libraries_by_name()
    found = set()
    for name, stage, arch in _shipped_bindings():
        binding = cake_jit.stage_binding(name, stage, arch)
        library = _library(name, stage, arch)
        twin_library = _library(name, stage[: -len(ACC)], arch)
        assert library in libraries and library in vendored, library
        entry = libraries[library]
        assert entry["arch"] == arch
        floor = _cuda_version(entry["min_cuda"])
        assert floor >= _cuda_version(libraries[twin_library]["min_cuda"]), library
        assert binding["specializations"].get("WGRAD_OUT") == 32, (name, stage)
        if stage != TAIL_REDUCE_ACC:
            assert floor == (13, 4), (library, entry["min_cuda"])
        found.add(library)
    # The tail reduce's build is one library per architecture and k tile.
    assert len(found) == len(_shipped_bindings())


def _nondet_tripwires(monkeypatch):
    """Replace the non-deterministic package's weight gradient and autograd wrapper by
    functions that fail the test when called: nothing may route there."""
    nondet_pkg = pytest.importorskip(
        "flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train"
    )
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        cake_autograd as nondet_autograd,
    )

    def tripwire(*args, **kwargs):
        raise AssertionError("the facade routed to the non-deterministic package")

    monkeypatch.setattr(
        nondet_pkg, "cake_moe_grouped_gemm_nondet_train_wgrad", tripwire
    )
    monkeypatch.setattr(nondet_autograd, "cake_moe_grouped_gemm_nondet", tripwire)


def test_facade_accumulate_rows(monkeypatch):
    """The facade hands ``accumulate`` and ``fuse_wgrad_accumulation`` to the package's
    functions of the same name with the same arguments, also under torch's deterministic
    mode: no reroute to the non-deterministic package, no warning; the prepared form
    hands ``accumulate`` to the backend's preparation."""
    facade = pytest.importorskip("flashinfer.cake_moe_grouped_gemm_det_train")
    _nondet_tripwires(monkeypatch)
    log = []

    def recorder(label):
        def record(*args, **kwargs):
            log.append((label, kwargs))
            return label

        return record

    monkeypatch.setattr(pkg, "cake_moe_grouped_gemm_det_train_wgrad", recorder("wgrad"))
    monkeypatch.setattr(
        cb, "cake_moe_grouped_gemm_det_train_prepare_wgrad", recorder("prepare")
    )
    monkeypatch.setattr(cake_autograd, "cake_moe_grouped_gemm_det", recorder("mm"))
    g, x, offs, out = _operands([2, 2], 256, 64, torch.float32, torch.device("cpu"))
    weight = torch.zeros(2, 256, 64, dtype=torch.bfloat16)
    for mode in (contextlib.nullcontext(), base._torch_deterministic()):
        with mode, warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert (
                facade.cake_moe_grouped_gemm_det_train_wgrad(
                    g, x, offs, out=out, accumulate=True
                )
                == "wgrad"
            )
            assert log.pop() == (
                "wgrad",
                dict(out=out, out_dtype=None, num_groups=None, accumulate=True),
            )
            assert (
                facade.cake_moe_grouped_gemm_det_train_prepare_wgrad(
                    g, x, offs, out=out, accumulate=True
                )
                == "prepare"
            )
            assert log.pop()[1]["accumulate"] is True
            assert (
                facade.cake_moe_grouped_gemm_det(
                    x, weight, offs, fuse_wgrad_accumulation=True
                )
                == "mm"
            )
            assert log.pop() == ("mm", dict(fuse_wgrad_accumulation=True))
            assert log == []
        assert [r for r in rec if r.category is UserWarning] == []


def _reject_cases(sizes, n, k):
    """``(label, out, out_dtype, operands override, error pattern)`` of the refused
    accumulate calls."""
    e, sum_m = len(sizes), sum(sizes)
    f32 = torch.float32
    cases = [
        ("out_none", None, None, {}, "pass the"),
        (
            "dtype_mismatch",
            torch.zeros(e, n, k, dtype=f32),
            torch.bfloat16,
            {},
            "out_dtype",
        ),
        (
            "fp16_out",
            torch.zeros(e, n, k, dtype=torch.float16),
            None,
            {},
            "bfloat16 or float32",
        ),
        (
            "two_dims",
            torch.zeros(e * n, k, dtype=f32),
            None,
            {},
            "unit stride along K",
        ),
        (
            "k_strided",
            torch.zeros(e, n, 2 * k, dtype=f32)[:, :, ::2],
            None,
            {},
            "unit stride along K",
        ),
        (
            "row_overlap",
            torch.zeros(e * n * k, dtype=f32).as_strided((e, n, k), (n * k, k // 2, 1)),
            None,
            {},
            "row stride",
        ),
        (
            "expert_overlap",
            torch.zeros(e * n * k, dtype=f32).as_strided((e, n, k), (n * k // 2, k, 1)),
            None,
            {},
            "expert stride",
        ),
        (
            "expanded",
            torch.zeros(1, n, k, dtype=f32).expand(e, n, k),
            None,
            {},
            "expert stride",
        ),
    ]
    # out sharing memory with g, x or offs.
    shared = torch.zeros(max(sum_m * n, sum_m * k) + e * n * k, dtype=torch.bfloat16)
    g_view = shared[: sum_m * n].view(sum_m, n)
    cases.append(
        (
            "overlaps_g",
            shared[sum_m * n // 2 :][: e * n * k].view(e, n, k),
            None,
            {"g": g_view},
            "with g",
        )
    )
    x_view = shared[: sum_m * k].view(sum_m, k)
    cases.append(
        ("overlaps_x", shared[: e * n * k].view(e, n, k), None, {"x": x_view}, "with x")
    )
    raw = torch.zeros(e * n * k * 4, dtype=torch.uint8)
    offs_view = raw[: 4 * e].view(torch.int32)
    cases.append(
        (
            "overlaps_offs",
            raw.view(torch.float32).view(e, n, k),
            None,
            {"offs": offs_view},
            "with offs",
        )
    )
    return cases


@pytest.mark.parametrize("tile", [256, 512])
def test_accumulate_refuses_unsafe_outputs(tile, monkeypatch):
    """``accumulate=True`` refuses (``ValueError``, before any workspace is allocated or
    program bound) a missing ``out``, an ``out_dtype`` other than ``out``'s, an ``out`` of
    another dtype or layout, rows or experts that overlap (an expanded ``out`` included)
    and an ``out`` that shares memory with ``g``, ``x`` or ``offs``, through the
    preparation and the one-shot function alike, on either k tile (the 256-wide one
    composes, the 512-wide one is fused for an fp32 ``out``: the refusals hold on both
    routes); and a non-bool ``accumulate``.  A row-padded, an expert-padded and a
    one-group ``out`` are accepted, on the route of the k tile."""
    arch = "sm_100a"
    _fake_device(monkeypatch, arch, 148)
    bound = _record_binds(monkeypatch)
    allocated = []
    real_bytes = cb._aligned_bytes
    monkeypatch.setattr(
        cb,
        "_aligned_bytes",
        lambda *args: allocated.append(args) or real_bytes(*args),
    )
    if tile == 512:
        monkeypatch.setitem(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"], arch, 1)
    sizes, n, k = [64, 0, 64], 256, 512
    g0, x0, offs0, _ = _operands(sizes, n, k, torch.float32, torch.device("cpu"))
    for label, out, out_dtype, override, pattern in _reject_cases(sizes, n, k):
        g, x, offs = (
            override.get("g", g0),
            override.get("x", x0),
            override.get("offs", offs0),
        )
        for call in (
            lambda: cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
                g, x, offs, out=out, out_dtype=out_dtype, accumulate=True
            ),
            lambda: pkg.cake_moe_grouped_gemm_det_train_wgrad(
                g, x, offs, out=out, out_dtype=out_dtype, accumulate=True
            ),
        ):
            with pytest.raises(ValueError, match=pattern):
                call()
        assert bound == [] and allocated == [], label
    _, _, _, out = _operands(sizes, n, k, torch.float32, torch.device("cpu"))
    for value in (1, 0, None, "True", torch.tensor(True)):
        for fn in (
            cb.cake_moe_grouped_gemm_det_train_prepare_wgrad,
            pkg.cake_moe_grouped_gemm_det_train_wgrad,
        ):
            with pytest.raises(ValueError, match="accumulate must be a bool"):
                fn(g0, x0, offs0, out=out, accumulate=value)
    assert bound == [] and allocated == []
    route = "composed" if tile == 256 else "fused"
    if route == "fused":
        assert _promised(arch, "main_halves", torch.float32)
    for accepted in (
        torch.zeros(3, n + 2, k + 32)[:, :n, :k],  # row and expert padding
        torch.zeros(3, n + 1, k)[:, :n],  # expert padding
        torch.zeros(3, n, k, requires_grad=True),  # a leaf that requires grad
    ):
        assert cb._accumulate_aligned(accepted)
        launch = cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g0, x0, offs0, out=accepted, accumulate=True
        )
        assert launch.out is accepted and launch.plan["accumulate"] == route
        assert launch.plan["tile_k"] == tile
    # One group: no expert pitch to check (an expert stride of 0 is accepted).
    g1, x1, offs1, _ = _operands([128], n, k, torch.float32, torch.device("cpu"))
    one = torch.zeros(n * k).as_strided((1, n, k), (0, k, 1))
    launch = cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g1, x1, offs1, out=one, accumulate=True
    )
    if tile == 512:  # 128 rows reach the forced 512-wide k tile: fused, in place
        assert launch.plan["accumulate"] == "fused" and launch.add_from is None
    else:
        assert launch.plan["accumulate"] == "composed"
        assert launch.add_from.shape == (1, n, k) and launch.add_from.is_contiguous()


@pytest.mark.parametrize("accumulate", [False, True])
def test_out_span_is_bounded_by_32_bit_addressing(accumulate, monkeypatch):
    """The kernels address ``out`` with 32-bit element offsets over its whole span, so the
    preparation bounds ``1 + sum((size - 1) * stride)``, not ``E`` times the expert stride: a
    one-group ``out`` with an expert stride of 0 and a row pitch of ``2 K`` whose span passes
    the limit is refused (``ValueError``, before any program is bound), as the overwrite and
    as the fused accumulate; at the limit it is accepted, unless its expert stride (which the
    binding takes as a 32-bit value but adds nothing to a one-group span) passes the limit."""
    arch = "sm_100a"
    _fake_device(monkeypatch, arch, 148)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"], arch, 1)
    bound = _record_binds(monkeypatch)
    n, k = 256, 512
    g, x, offs, _ = _operands([128], n, k, torch.float32, torch.device("cpu"))
    out = torch.zeros(n * 2 * k).as_strided((1, n, k), (0, 2 * k, 1))
    span = (n - 1) * 2 * k + k
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "int32_elems", span - 1)
    with pytest.raises(ValueError, match="exceeds the 32-bit element addressing"):
        cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out=out, accumulate=accumulate
        )
    assert bound == []
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "int32_elems", span)
    wide = out.as_strided(out.shape, (span + 1, 2 * k, 1))
    with pytest.raises(ValueError, match="exceeds the 32-bit element addressing"):
        cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out=wide, accumulate=accumulate
        )
    assert bound == []
    launch = cb.cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g, x, offs, out=out, accumulate=accumulate
    )
    assert launch.out is out
    assert launch.plan["accumulate"] == ("fused" if accumulate else "off")


def test_sum_m_zero_accumulates_nothing(monkeypatch):
    """No row: the accumulate preparation binds nothing, keeps no scratch, does not
    zero-fill ``out`` (the overwrite does) and its launch leaves ``out`` as it is."""
    _fake_device(monkeypatch, "sm_103a", 160)
    monkeypatch.setattr(
        cb, "tvm_ffi", types.SimpleNamespace(use_torch_stream=contextlib.nullcontext)
    )
    g, x, offs, out = _operands([0, 0], 256, 512, torch.float32, torch.device("cpu"))
    ovw = prepare_wgrad(g, x, offs, out=torch.ones_like(out))
    acc = prepare_wgrad(g, x, offs, out=out.fill_(3.0), accumulate=True)
    assert ovw.zero_fill and ovw.launches == () and ovw.plan["accumulate"] == "off"
    assert acc.launches == () and acc.stages == () and not acc.zero_fill
    assert acc.add_from is None and acc.plan["accumulate"] == "composed"
    assert not acc.bump_version
    version = out._version
    assert acc.launch() is out and bool((out == 3.0).all())
    assert out._version == version
    assert ovw.launch() is ovw.out and int(ovw.out.count_nonzero()) == 0
    # The 512-wide k tile forced: the route would be fused, and still nothing launches.
    monkeypatch.setitem(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"], "sm_103a", 0)
    fused = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert fused.plan["tile_k"] == 512 and fused.plan["accumulate"] == "fused"
    assert fused.launches == () and fused.stages == () and not fused.bump_version
    assert fused.launch() is out and bool((out == 3.0).all())
    assert out._version == version


def _recorded_launch(monkeypatch, out):
    """An accumulate ``CakeMoEGroupedGemmDetLaunch`` whose one program is a recorder that logs
    ``out`` as the program sees it and writes 2 into the scratch; returns ``(launch,
    log)``."""
    monkeypatch.setattr(
        cb, "tvm_ffi", types.SimpleNamespace(use_torch_stream=contextlib.nullcontext)
    )
    log = []
    scratch = torch.zeros(out.shape, dtype=out.dtype)

    def program(*args):
        log.append(out.detach().clone())
        scratch.fill_(2.0)

    launch = cb.CakeMoEGroupedGemmDetLaunch(
        op="wgrad",
        record_name="cake_moe_grouped_det_gemm_wgrad_f32_k512",
        stages=("main_halves",),
        plan={"accumulate": "composed"},
        out=out,
        launches=(cb._Launch(entry=program, arguments=()),),
        add_from=scratch,
    )
    return launch, log


def test_launch_adds_the_scratch_after_the_programs(monkeypatch):
    """Each ``launch()`` (and ``__call__``) runs the programs, then adds the scratch into
    ``out`` once (not idempotent), bumping ``out._version`` and recording no autograd
    node, also for a leaf that requires grad; an inference tensor is added into outside
    inference mode; torch's deterministic mode (strict and ``warn_only``) raises no alert
    and no warning."""
    out = torch.zeros(4, requires_grad=True)
    launch, log = _recorded_launch(monkeypatch, out)
    version = out._version
    assert launch.launch() is out
    launch()
    assert [t.tolist() for t in log] == [[0.0] * 4, [2.0] * 4]
    assert out.tolist() == [4.0] * 4 and out.grad_fn is None
    assert out._version == version + 2
    with torch.inference_mode():
        frozen = torch.zeros(4)
    assert frozen.is_inference()
    inference, _ = _recorded_launch(monkeypatch, frozen)
    inference.launch()
    assert frozen.tolist() == [2.0] * 4
    plain, _ = _recorded_launch(monkeypatch, torch.zeros(4))
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        for warn_only in (False, True):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                plain.launch()
            assert [w for w in rec if w.category is UserWarning] == [], warn_only
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    assert plain.out.tolist() == [4.0] * 4


def test_fused_launch_bumps_the_version_without_an_add(monkeypatch):
    """A fused accumulate launch (``bump_version``, no scratch) runs its programs, adds
    nothing on the host and bumps ``out._version`` once per ``launch()`` (and
    ``__call__``) with no autograd node, also for a leaf that requires grad; an inference
    tensor is launched into without a bump; torch's deterministic mode (strict and
    ``warn_only``) raises no alert and no warning."""
    monkeypatch.setattr(
        cb, "tvm_ffi", types.SimpleNamespace(use_torch_stream=contextlib.nullcontext)
    )

    def fused(out, log):
        def program(*args):
            log.append(out.detach().clone())
            # The programs add in place, outside torch: through ``.data``, whose writes
            # bump no version counter of ``out``.
            mode = (
                torch.inference_mode if out.is_inference() else contextlib.nullcontext
            )
            with mode():
                out.data.add_(2.0)

        return cb.CakeMoEGroupedGemmDetLaunch(
            op="wgrad",
            record_name=F32_K512,
            stages=("main_halves_acc", TAIL_REDUCE_ACC),
            plan={"accumulate": "fused"},
            out=out,
            launches=(cb._Launch(entry=program, arguments=()),),
            bump_version=True,
        )

    out = torch.zeros(4, requires_grad=True)
    log = []
    launch = fused(out, log)
    version = out._version
    assert launch.launch() is out
    assert out._version == version + 1
    launch()
    assert [t.tolist() for t in log] == [[0.0] * 4, [2.0] * 4]
    assert out.tolist() == [4.0] * 4 and out.grad_fn is None
    assert out._version == version + 2
    with torch.inference_mode():
        frozen = torch.zeros(4)
    inference = fused(frozen, [])
    inference.launch()
    assert frozen.tolist() == [2.0] * 4
    plain = fused(torch.zeros(4), [])
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        for warn_only in (False, True):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                plain.launch()
            assert [w for w in rec if w.category is UserWarning] == [], warn_only
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    assert plain.out.tolist() == [4.0] * 4


def test_fuse_flag_refusals_on_the_host():
    """``fuse_wgrad_accumulation`` must be a bool (``ValueError``, also as the fifth
    argument of ``apply``, whose fourth stays the internal slot: ``TypeError``); with it,
    a ``w`` that requires grad needs a ``main_grad`` tensor of its shape, fp32 or bf16, on
    its device, with unit stride along the last dimension (``ValueError`` at the forward
    call, before any launch); a row-padded ``main_grad`` is accepted; a ``w`` that does
    not require grad needs none, and neither does a call under ``torch.no_grad()`` or
    inference mode, which autograd does not record."""
    e, n, k = 2, 256, 64
    x = torch.zeros(4, k, dtype=torch.bfloat16)
    offs = torch.tensor([2, 4], dtype=torch.int32)

    def call(w, flag=True):
        return pkg.cake_moe_grouped_gemm_det(x, w, offs, fuse_wgrad_accumulation=flag)

    w = torch.zeros(e, n, k, dtype=torch.bfloat16, requires_grad=True)
    for value in (1, None, "True"):
        with pytest.raises(ValueError, match="fuse_wgrad_accumulation must be a bool"):
            call(w, value)
        with pytest.raises(ValueError, match="fuse_wgrad_accumulation must be a bool"):
            cake_autograd.CakeMoEGroupedGemmDetFunction.apply(x, w, offs, None, value)
        with pytest.raises(ValueError, match="fuse_wgrad_accumulation must be a bool"):
            cake_autograd.CakeMoEGroupedGemmDetFunction.apply(
                x, w, offs, cake_autograd._DEFER_FWD, value
            )
    for stale in (True, False):
        with pytest.raises(TypeError, match="takes no fourth argument"):
            cake_autograd.CakeMoEGroupedGemmDetFunction.apply(x, w, offs, stale, True)
    with pytest.raises(TypeError):
        pkg.cake_moe_grouped_gemm_det(x, w, offs, True)
    with pytest.raises(ValueError, match="no main_grad"):
        call(w)
    for bad in (
        torch.zeros(e, n, k + 1),
        torch.zeros(e, n, k, dtype=torch.float16),
        torch.zeros(e, n, 2 * k)[:, :, ::2],
        torch.zeros(e, n, k, device="meta"),
        torch.zeros(e * n * k),
    ):
        w.main_grad = bad
        with pytest.raises(ValueError, match="w.main_grad must be"):
            call(w)
    w.main_grad = torch.zeros(e, n, k + 32)[:, :, :k]  # row-padded: accepted
    with pytest.raises(ValueError) as info:  # CPU operands: the device check
        call(w)
    assert "main_grad" not in str(info.value)
    frozen = torch.zeros(e, n, k, dtype=torch.bfloat16)
    with pytest.raises(ValueError) as info:  # the device check, not main_grad
        call(frozen)
    assert "main_grad" not in str(info.value)
    bare = torch.zeros(e, n, k, dtype=torch.bfloat16, requires_grad=True)
    for mode in (torch.no_grad, torch.inference_mode):
        with mode(), pytest.raises(ValueError) as info:  # the device check again
            call(bare)
        assert "main_grad" not in str(info.value), mode.__name__


@pytest.mark.parametrize("mode", ["trace", "compile"])
def test_fuse_flag_is_refused_under_trace_and_compile(mode, monkeypatch):
    """``fuse_wgrad_accumulation=True`` raises ``NotImplementedError`` before ``apply``
    under ``torch.jit.trace`` and ``torch.compile``."""
    if mode == "trace":
        monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    else:
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    w = torch.zeros(2, 256, 64, dtype=torch.bfloat16, requires_grad=True)
    w.main_grad = torch.zeros(2, 256, 64)
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    offs = torch.tensor([2, 4], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="fuse_wgrad_accumulation"):
        pkg.cake_moe_grouped_gemm_det(x, w, offs, fuse_wgrad_accumulation=True)


# ---------------------------------------------------------------------------
# The autograd flag's contract with CPU stand-ins for the preparations
# ---------------------------------------------------------------------------


class _HostPreparations:
    """CPU stand-ins for the three preparations the autograd wrapper calls (exact on the
    small-integer operands of these tests): the forward, the data gradient, and the
    weight gradient written over ``out`` or, with ``accumulate=True``, in ``out``'s dtype
    and added into it with ``out.add_`` (the package's composed accumulate).  Every call is
    logged as ``(op, kwargs)``; ``torch.cuda.current_stream`` and
    ``torch.cuda.is_current_stream_capturing`` are stubs."""

    def __init__(self, monkeypatch):
        self.log = []
        monkeypatch.setattr(
            cake_autograd, "cake_moe_grouped_gemm_det_train_prepare_forward", self.fwd
        )
        monkeypatch.setattr(
            cake_autograd, "cake_moe_grouped_gemm_det_train_prepare_dgrad", self.dgrad
        )
        monkeypatch.setattr(
            cake_autograd, "cake_moe_grouped_gemm_det_train_prepare_wgrad", self.wgrad
        )
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda device=None: types.SimpleNamespace(query=lambda: True),
        )
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def ops(self, op):
        return [kwargs for name, kwargs in self.log if name == op]

    @staticmethod
    def _bounds(offs, groups):
        ends = [int(v) for v in offs.tolist()]
        if len(ends) == groups + 1:
            ends = ends[1:]
        return list(zip([0] + ends[:-1], ends, strict=True))

    @staticmethod
    def _launch(run):
        return types.SimpleNamespace(launch=run)

    def fwd(self, x, w, offs, out=None):
        self.log.append(("fwd", {}))

        def run():
            for e, (s, t) in enumerate(self._bounds(offs, w.shape[0])):
                out[s:t] = (x[s:t].float() @ w[e].float().T).to(out.dtype)
            return out

        return self._launch(run)

    def dgrad(self, g, w, offs, out=None):
        self.log.append(("dgrad", {}))

        def run():
            for e, (s, t) in enumerate(self._bounds(offs, w.shape[0])):
                out[s:t] = (g[s:t].float() @ w[e].float()).to(out.dtype)
            return out

        return self._launch(run)

    def wgrad(self, g, x, offs, out=None, out_dtype=None, num_groups=None, **kw):
        self.log.append(("wgrad", dict(out=out, num_groups=num_groups, **kw)))

        def run():
            dw = _host_dw(g, x, self._bounds(offs, num_groups), out.shape)
            with torch.no_grad():
                if kw.get("accumulate", False):
                    out.add_(dw.to(out.dtype))
                else:
                    out.copy_(dw)
            return out

        return self._launch(run)


def _host_dw(g, x, bounds, shape):
    dw = torch.zeros(shape, dtype=torch.float32)
    for e, (s, t) in enumerate(bounds):
        dw[e] = g[s:t].float().T @ x[s:t].float()
    return dw


HOST_SIZES = [3, 0, 2]  # an empty group
HOST_N, HOST_K = 4, 6


def _host_case(seed, main_grad_dtype=torch.float32):
    """Small-integer operands (every product and sum exact in fp32 and in bf16):
    ``(x0, w0, g, offs, mg0, dW)`` with ``dW`` the fp32 weight gradient of ``g``."""
    gen = torch.Generator().manual_seed(seed)
    e, sum_m = len(HOST_SIZES), sum(HOST_SIZES)

    def ints(*shape):
        return torch.randint(-2, 3, shape, generator=gen).to(torch.bfloat16)

    x0, w0, g = ints(sum_m, HOST_K), ints(e, HOST_N, HOST_K), ints(sum_m, HOST_N)
    offs = torch.tensor(HOST_SIZES, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    mg0 = torch.randint(-8, 9, (e, HOST_N, HOST_K), generator=gen).to(main_grad_dtype)
    bounds = _HostPreparations._bounds(offs, e)
    return x0, w0, g, offs, mg0, _host_dw(g, x0, bounds, (e, HOST_N, HOST_K))


def _host_gemm(offs, fuse=True):
    return lambda a, b: pkg.cake_moe_grouped_gemm_det(
        a, b, offs, fuse_wgrad_accumulation=fuse
    )


def _host_leaves(x0, w0, mg0, main_grad=True, requires_grad=True, **attributes):
    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(requires_grad)
    if main_grad:
        w.main_grad = mg0.clone()
    for key, value in attributes.items():
        setattr(w, key, value)
    return x, w


@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
def test_autograd_fuse_contract_on_the_host(main_grad_dtype, monkeypatch):
    """With CPU stand-ins for the preparations: the backward hands ``w.main_grad`` (the
    tensor itself, read at backward time) to the weight-gradient preparation with
    ``accumulate=True`` and ``num_groups=E``, adds once per backward (two micro-steps add
    twice, ``retain_graph`` twice adds twice, ``torch.autograd.grad`` over ``x`` alone still
    adds), returns no gradient for ``w`` without ``grad_added_to_main_grad`` and the fresh
    placeholder with it (zeros under ``zero_out_wgrad``), leaves ``x.grad`` as without the
    flag, never adds for a frozen ``w``, honours a re-pointed ``main_grad``, refuses a
    wrong one at the backward and a ``create_graph`` backward, runs a direct ``apply``
    (checked at the backward only), and under torch's deterministic mode warns nothing."""
    host = _HostPreparations(monkeypatch)
    x0, w0, g, offs, mg0, dw = _host_case(1, main_grad_dtype)
    step = dw.to(main_grad_dtype)
    gemm = _host_gemm(offs)

    x, w = _host_leaves(x0, w0, mg0)
    main_grad = w.main_grad
    y = gemm(x, w)
    y.backward(g)
    dx = x.grad.clone()
    gemm(x0.clone().requires_grad_(True), w).backward(g)  # a second micro-step
    assert w.grad is None and w.main_grad is main_grad
    assert torch.equal(w.main_grad, mg0.clone().add_(step).add_(step))
    [first, second] = host.ops("wgrad")
    assert first["accumulate"] is True and first["out"] is main_grad
    assert first["num_groups"] == len(HOST_SIZES) and second["out"] is main_grad
    x, w = _host_leaves(x0, w0, mg0)
    y_plain = _host_gemm(offs, fuse=False)(x, w)
    y_plain.backward(g)
    assert torch.equal(y.detach(), y_plain.detach())
    assert torch.equal(x.grad, dx) and torch.equal(w.grad, dw.to(w.dtype))
    assert torch.equal(w.main_grad, mg0)  # the flag off never touches main_grad
    assert host.ops("wgrad")[-1].get("accumulate", False) is False

    for zero_out in (False, True):
        x, w = _host_leaves(
            x0, w0, mg0, grad_added_to_main_grad=False, zero_out_wgrad=zero_out
        )
        y = gemm(x, w)
        assert torch.equal(y.detach(), y_plain.detach())
        y.backward(g)
        assert w.grad_added_to_main_grad is True
        assert w.grad.shape == w0.shape and w.grad.dtype == w0.dtype
        if zero_out:
            assert int(w.grad.count_nonzero()) == 0
        assert torch.equal(w.main_grad, mg0.clone().add_(step))

    x, w = _host_leaves(x0, w0, mg0)
    y = gemm(x, w)
    y.backward(g, retain_graph=True)
    y.backward(g)
    assert torch.equal(w.main_grad, mg0.clone().add_(step).add_(step))

    x, w = _host_leaves(x0, w0, mg0)
    (dx_only,) = torch.autograd.grad(gemm(x, w), (x,), g)
    assert torch.equal(w.main_grad, mg0.clone().add_(step)) and w.grad is None
    assert torch.equal(dx_only, dx)

    calls = len(host.ops("wgrad"))
    x, w = _host_leaves(x0, w0, mg0, requires_grad=False)
    gemm(x, w).backward(g)
    assert torch.equal(w.main_grad, mg0) and x.grad is not None
    assert len(host.ops("wgrad")) == calls

    x, w = _host_leaves(x0, w0, mg0)
    y = gemm(x, w)
    moved = torch.zeros_like(mg0)
    w.main_grad = moved
    y.backward(g)
    assert torch.equal(moved, step) and w.main_grad is moved

    x, w = _host_leaves(x0, w0, mg0)
    y = gemm(x, w)
    w.main_grad = torch.zeros(1, dtype=main_grad_dtype)
    with pytest.raises(ValueError, match="w.main_grad must be"):
        y.backward(g)

    x, w = _host_leaves(x0, w0, mg0)
    with pytest.raises(NotImplementedError, match="create_graph"):
        torch.autograd.grad(gemm(x, w), (x, w), g, create_graph=True)

    x, w = _host_leaves(x0, w0, mg0)
    y = cake_autograd.CakeMoEGroupedGemmDetFunction.apply(x, w, offs, None, True)
    y.backward(g)
    assert torch.equal(w.main_grad, mg0.clone().add_(step)) and w.grad is None
    x, w = _host_leaves(x0, w0, mg0, main_grad=False)
    y = cake_autograd.CakeMoEGroupedGemmDetFunction.apply(x, w, offs, None, True)
    with pytest.raises(ValueError, match="no main_grad"):
        y.backward(g)

    x, w = _host_leaves(x0, w0, mg0)
    with base._torch_deterministic(), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gemm(x, w).backward(g)
    assert [r for r in rec if r.category is UserWarning] == []
    assert torch.equal(w.main_grad, mg0.clone().add_(step))


@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
@pytest.mark.parametrize("ckpt", ["non_reentrant", "nested", "reentrant"])
def test_autograd_fuse_under_checkpoints_on_the_host(ckpt, early_stop, monkeypatch):
    """With CPU stand-ins: under a non-reentrant checkpoint (also nested) the backward
    adds once into ``w.main_grad`` with the bits of the plain call, the recompute needs
    no ``main_grad`` check, and the recomputed forward launches only without early stop;
    under a reentrant checkpoint ``w`` arrives detached without ``main_grad``
    (``ValueError``, use ``use_reentrant=False``)."""
    host = _HostPreparations(monkeypatch)
    x0, w0, g, offs, mg0, dw = _host_case(2)
    gemm = _host_gemm(offs)
    x, w = _host_leaves(x0, w0, mg0)
    with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
        if ckpt == "reentrant":
            y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=True)
            with pytest.raises(ValueError, match="main_grad"):
                y.backward(g)
            return
        if ckpt == "nested":

            def inner(a, b):
                return torch.utils.checkpoint.checkpoint(
                    gemm, a, b, use_reentrant=False
                )

            y = torch.utils.checkpoint.checkpoint(inner, x, w, use_reentrant=False)
        else:
            y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
        before = len(host.ops("fwd"))
        y.backward(g)
    recomputed = len(host.ops("fwd")) - before
    assert torch.equal(w.main_grad, mg0.clone().add_(dw)) and w.grad is None
    if ckpt == "non_reentrant":
        assert recomputed == (0 if early_stop else 1)
    x2, w2 = _host_leaves(x0, w0, mg0)
    gemm(x2, w2).backward(g)
    assert torch.equal(x.grad, x2.grad) and torch.equal(w.main_grad, w2.main_grad)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capturing"])
def test_backward_never_queries_a_captured_stream(capturing, monkeypatch):
    """The backward binds the CUDA context of its thread with one runtime call: the
    capture-status query, then a stream query only outside a CUDA graph capture (querying
    a stream that is being captured ends the capture with an error), with and without the
    fuse flag."""
    _HostPreparations(monkeypatch)
    queries = []
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: types.SimpleNamespace(
            query=lambda: queries.append(device) or True
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    x0, w0, g, offs, mg0, _dw = _host_case(7)
    for fuse in (False, True):
        x, w = _host_leaves(x0, w0, mg0)
        _host_gemm(offs, fuse)(x, w).backward(g)
    assert len(queries) == (0 if capturing else 2), queries


@pytest.mark.parametrize("layout", ["expanded", "row_overlap"])
def test_overlapping_main_grad_is_refused_at_the_forward(layout, monkeypatch):
    """A ``main_grad`` whose experts or rows overlap (an expanded one, or a row stride
    below ``K``) is refused with ``ValueError`` at a forward call that autograd records,
    before anything launches, as at the backward."""
    host = _HostPreparations(monkeypatch)
    x0, w0, g, offs, mg0, _dw = _host_case(8)
    e, n, k = w0.shape
    if layout == "expanded":
        main_grad = torch.zeros(1, n, k).expand(e, n, k)
    else:
        main_grad = torch.zeros(e * n * k).as_strided(
            (e, n, k), (n * (k - 1), k - 1, 1)
        )
    x, w = _host_leaves(x0, w0, mg0)
    w.main_grad = main_grad
    with pytest.raises(ValueError, match="must not overlap"):
        _host_gemm(offs)(x, w)
    assert host.log == []


def _arch(device):
    return SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]


def _bits(t):
    return t.view(INT_VIEW[t.dtype])


def _assert_bitwise(out, want, what=""):
    """``out`` and ``want`` have the same bits, NaNs aside (both NaN, any payload)."""
    nan = torch.isnan(out)
    assert torch.equal(nan, torch.isnan(want)), f"{what}: NaN positions differ"
    differ = (_bits(out) != _bits(want)) & ~nan
    count = int(differ.sum())
    assert count == 0, (
        f"{what}: {count} elements differ from the oracle, e.g. {differ.nonzero()[:4].tolist()}"
    )


def _rms(t):
    return float(t.double().pow(2).mean().sqrt().clamp_min(1e-30))


def _out0(kind, dw, sizes, dtype, gen):
    """The starting ``out`` of a case: ``"zeros"``, ``"randn"`` (randn x rms(dW)) or
    ``"specials"`` (randn x rms(dW) with -0.0, NaN and +-max/2^8 at tile and group
    boundaries); empty groups hold random bits with -0.0, a NaN payload and a subnormal
    in every case."""
    e, n, k = dw.shape
    scale = _rms(dw)
    if kind == "zeros":
        out0 = torch.zeros(e, n, k, device=dw.device)
    else:
        out0 = torch.randn(e, n, k, generator=gen, device=dw.device) * scale
    out0 = out0.to(dtype)
    if kind == "specials":
        big = torch.finfo(dtype).max / 2.0**8
        for group, rows in enumerate(sizes):
            if not rows:
                continue
            for (r, c), value in (
                ((0, 0), -0.0),
                ((0, 2), float("nan")),
                ((min(127, n - 1), min(128, k - 1)), big),
                ((min(128, n - 1), min(127, k - 1)), -big),
                ((n - 1, k - 1), -0.0),
            ):
                out0[group, r, c] = value
    nan_bits = 0x7FC01234 if dtype == torch.float32 else 0x7FC1
    subnormal = torch.tensor(FP32_SUBNORMAL, dtype=dtype)
    for group, rows in enumerate(sizes):
        if rows:
            continue
        noise = torch.randint(
            -(1 << 15), 1 << 15, (n, k), generator=gen, device=dw.device
        )
        _bits(out0[group]).copy_(noise.to(INT_VIEW[dtype]))
        out0[group, 0, 0] = -0.0
        _bits(out0[group])[0, 1] = nan_bits - (1 << 32 if nan_bits >= 1 << 31 else 0)
        out0[group, 1, 0] = subnormal
    return out0


def _overwrite(g, x, offs, out_dtype, num_groups=None):
    """The deterministic overwrite weight gradient of the call in ``out_dtype``."""
    return (
        prepare_wgrad(g, x, offs, out_dtype=out_dtype, num_groups=num_groups)
        .launch()
        .clone()
    )


def _steps_expected(out0, dws):
    """``(ref, scale)`` of ``out0`` plus each fp64 weight gradient of ``dws`` added in
    turn: the fp64 sum (as fp32), and the magnitude the bf16 budget scales with, one term
    per rounding: ``sum_i (|dW_i| + |out0 + dW_1 + ... + dW_i|)`` (each step rounds its
    ``dW`` and its partial sum once)."""
    total = out0.double()
    scale = torch.zeros_like(total)
    for dw in dws:
        total = total + dw
        scale = scale + dw.abs() + total.abs()
    return total.float(), scale.float()


def _expected(out0, g, x, sizes, times=1):
    """``(ref, scale)`` of ``out0`` plus ``times`` adds of ``sum_e G[e].T X[e]``
    (``_steps_expected``)."""
    dw64 = base._reference("wgrad", g, x, sizes).double()
    return _steps_expected(out0, [dw64] * times)


def _assert_within_budget(out, expected, mask=None):
    """The budget of ``out``'s dtype against ``expected`` over ``mask`` (default all):
    fp32 rel-L2 against the fp64 reference (the deterministic ``dW``'s budget plus one fp32
    rounding); bf16 every element within ``1e-2 + 1e-2 * scale`` (``_steps_expected``:
    one term per bf16 rounding)."""
    ref, scale = expected
    if mask is not None:
        out, ref, scale = out[mask], ref[mask], scale[mask]
    if out.dtype == torch.bfloat16:
        assert bool(torch.isfinite(out).all()), "a non-finite bf16 result"
        err = (out.double() - ref.double()).abs()
        bound = base.ATOL + base.RTOL * scale.double()
        bad = int((err > bound).sum())
        assert bad == 0, f"bf16: {bad} elements beyond the operand-scaled budget"
        return
    rel_l2 = float(
        (out.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )
    assert rel_l2 <= base.FP32_REL_L2, f"fp32 rel-L2 {rel_l2:.3e}"


def _ordinary_mask(out0, sizes):
    """Elements of groups with rows whose ``out0`` is ordinary (no NaN, no +-max/2^8)."""
    mask = torch.isfinite(out0) & (out0.abs() < torch.finfo(out0.dtype).max / 2.0**9)
    for group, rows in enumerate(sizes):
        if not rows:
            mask[group] = False
    return mask


def _assert_empty_groups_keep_their_values(out, out0, sizes):
    for group, rows in enumerate(sizes):
        if rows:
            continue
        same = (out[group] == out0[group]) | (
            torch.isnan(out[group]) & torch.isnan(out0[group])
        )
        assert bool(same.all()), group


def _device_case(device, form, split, monkeypatch, out_dtype=torch.float32):
    """Force ``form`` on the device and return ``(sizes, n, k)`` of a shape that takes it
    with a split tail or without; skips where the architecture has no such form, fails
    where it has the form but no candidate shape (``test_forced_form_cases_exist`` covers
    the SM counts of the parts)."""
    arch = _arch(device)
    tile, stage = FORMS[form]
    if not _force_form(monkeypatch, form, arch):
        pytest.skip(f"no {form} weight gradient on {arch}")
    name = cake_jit.record_name("wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=tile)
    if not cake_jit.stage_served(name, stage, arch):
        pytest.skip(f"route {name} serves no {stage} stage on {arch}")
    case = _find_case(arch, cb.sm_count(device), form, split, out_dtype)
    assert case is not None, (form, split, arch, cb.sm_count(device))
    return case


def _device_route(device, plan, stage, out_dtype):
    """The route an accumulate of the overwrite ``plan`` / main ``stage`` takes on the device:
    ``"fused"`` where this release ships the stage's accumulate build for ``out_dtype`` on
    the architecture (and so the tail reduce's), else ``"composed"``.  A shipped build whose
    library cannot load here (neither prebuilt nor a JIT toolkit at its ``min_cuda``)
    skips with "cannot load" in the reason: that call composes (``"min_cuda"``), which the
    CPU tests and the composed GPU cases cover."""
    arch = _arch(device)
    if plan["tile_k"] != 512 or not _promised(arch, stage, out_dtype):
        return "composed"
    name = cake_jit.record_name("wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=512)
    builds = [(name, stage + ACC)]
    tail = _tail_route(plan, out_dtype)
    if tail is not None:
        builds.append((tail, TAIL_REDUCE_ACC))
    for route, build in builds:
        if not cb.acc_library_loadable(route, build, arch):
            pytest.skip(
                f"the {build} library of {route} cannot load here (not prebuilt, and "
                "FlashInfer's JIT cannot build it at its build.json min_cuda): the call composes"
            )
    return "fused"


def _force_composed(monkeypatch):
    """Make every accumulate build count as not loadable, so a call composes
    (``"min_cuda"``) with the overwrite's programs, without the once-per-process warning."""
    monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: False)
    monkeypatch.setattr(cb, "_WARNED", {"accumulate_min_cuda"})


def _route_case(device, route, split, monkeypatch):
    """``(sizes, n, k)`` of a ``main_halves`` shape with or without a split tail whose fp32
    accumulate takes ``route`` (``"composed"`` forced by ``_force_composed``)."""
    sizes, n, k = _device_case(device, "main_halves", split, monkeypatch)
    if route == "composed":
        _force_composed(monkeypatch)
    else:
        _stage, plan = _det_plan(sizes, n, k, _arch(device), cb.sm_count(device))
        assert _device_route(device, plan, "main_halves", torch.float32) == "fused"
    return sizes, n, k


# ---------------------------------------------------------------------------
# The accumulate launch (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "split_tail"])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("form", list(FORMS))
def test_accumulate_is_the_overwrite_plus_add(form, out_dtype, split, monkeypatch):
    """Every weight-gradient form of the device, with and without the ordered split
    tail, into an fp32 and a bf16 ``out`` holding specials: the accumulate launch takes
    the overwrite's plan and route (fused where the release ships the form's accumulate
    build for the dtype: the overwrite's stages with ``_acc``; else composed: the
    overwrite's stages) and gives bitwise ``out0.add_(dW)`` (AccumulateGrad of the
    deterministic ``dW`` in ``out``'s dtype; NaN payloads aside), the same bits at each of
    ``REPEATS`` launches from fresh copies of ``out0`` (run to run; every other launch from
    a fresh preparation), within the budget of the fp64 ``out0 + dW``; empty groups keep
    their values."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _device_case(device, form, split, monkeypatch, out_dtype)
    _tile, stage = FORMS[form]
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=31)
    ovw = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    dw = ovw.launch().clone()
    gen = torch.Generator(device=device).manual_seed(32)
    out0 = _out0("specials", dw.float(), sizes, out_dtype, gen)
    out = out0.clone()
    acc = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    route = _device_route(device, ovw.plan, stage, out_dtype)
    assert acc.plan["accumulate"] == route, acc.plan
    suffix = ACC if route == "fused" else ""
    assert ovw.stages[0] == stage
    assert acc.stages == tuple(name + suffix for name in ovw.stages)
    assert (acc.plan["tail_splits"] > 1) == split
    assert (TAIL_REDUCE + suffix in acc.stages) == split
    assert _plan_without(acc.plan, "accumulate", "accumulate_reason") == _plan_without(
        ovw.plan, "accumulate"
    )
    oracle = out0.clone().add_(dw)
    first = None
    for repeat in range(REPEATS):
        out.copy_(out0)
        if repeat % 2:
            prepare_wgrad(g, x, offs, out=out, accumulate=True).launch()
        else:
            acc.launch()
        if first is None:
            torch.cuda.synchronize()
            _assert_bitwise(out, oracle, f"{form} {out_dtype} {route}")
            first = out.clone()
        else:
            assert torch.equal(_bits(out), _bits(first)), "differs run to run"
    torch.cuda.synchronize()
    payloads = int(((_bits(first) != _bits(oracle)) & torch.isnan(first)).sum())
    print(
        f"ACCFORM form={form} dtype={DTYPE_NAMES[out_dtype]} split={split} route={route} "
        f"stages={','.join(acc.stages)} n={n} k={k} nan_payload_diffs={payloads}"
    )
    _assert_within_budget(
        out, _expected(out0, g, x, sizes), _ordinary_mask(out0, sizes)
    )
    _assert_empty_groups_keep_their_values(out, out0, sizes)
    one_shot = out0.clone()
    assert pkg.cake_moe_grouped_gemm_det_train_wgrad(
        g, x, offs, out=one_shot, accumulate=True
    ) is (one_shot)
    torch.cuda.synchronize()
    assert torch.equal(_bits(one_shot), _bits(first))


# The fused-against-composed cases of each half-release form: a forced shape with an
# unsplit and with a split tail (``CASE_SIZES``), the planned walk with its trailing empty
# groups and a split tail beside empty groups (``EMPTY_TAIL_SIZES``), and 32 groups.
FUSED_CASES = ("unsplit", "split_tail", "empty_last", "empty_split", "e32")


def _shape_of(arch, sms, form, sizes):
    """``(n, k)``: the first ``K`` of ``_form_ks(form)`` and ``N`` up to 48 column blocks
    whose plan on ``arch`` takes ``form`` for ``sizes``, preferring a split tail; None when
    no candidate takes the form.  The constants must force ``form`` already."""
    tile, stage = FORMS[form]
    span = int(HOST_PLAN_CONSTANTS["int32_elems"])
    unsplit = None
    for k in _form_ks(form):
        for blocks in range(1, 49):
            n = 256 * blocks
            if len(sizes) * (n + 2) * (k + 32) > span:
                break
            got, plan = _det_plan(sizes, n, k, arch, sms)
            if got != stage or plan["tile_k"] != tile:
                break
            if plan["tail_splits"] > 1:
                return n, k
            unsplit = unsplit or (n, k)
    return unsplit


def _planted_out0(dw, sizes, gen):
    """An fp32 ``out0`` with no NaN for the bitwise comparison of the two routes: randn x
    rms(dW); every empty group all ``-0.0`` but for a positive and a negative subnormal;
    in every group with rows ``-0.0`` at two corners, subnormals of both signs down column
    1 (where ``dW`` is zero, ``_zero_column``) and at four scattered places."""
    e, n, k = dw.shape
    out0 = torch.randn(e, n, k, generator=gen, device=dw.device) * _rms(dw)
    sub = FP32_SUBNORMAL
    for group, rows in enumerate(sizes):
        if not rows:
            out0[group].fill_(-0.0)
            out0[group, 0, 1] = sub
            out0[group, 1, 0] = -sub
            continue
        out0[group, 0, 0] = -0.0
        out0[group, n - 1, k - 1] = -0.0
        out0[group, 0::2, 1] = sub
        out0[group, 1::2, 1] = -sub
        for r, c in ((1, 0), (n // 2, k // 2), (n - 1, 0), (min(255, n - 1), 256 % k)):
            out0[group, r, c] = sub if (r + c) % 2 else -sub
    return out0


def _zero_column(x):
    """``x`` with column 1 zero, so every ``dW[e][:, 1]`` is a zero."""
    x[:, 1] = 0
    return x


@pytest.mark.parametrize("case", FUSED_CASES)
@pytest.mark.parametrize("form", TWIN_FORMS)
def test_fused_equals_composed_bitwise(form, case, monkeypatch):
    """The fused route of every half-release form the device ships, against the composed
    route of the same call (forced: the accumulate builds counted as not loadable) and the
    oracle ``out0.add_(dW32)``: bitwise equal (int32 view, no NaN in ``out0``) on a split
    and an unsplit tail, the planned walk with empty groups last, a split tail beside empty
    groups, and 32 groups; ``-0.0`` in an empty group becomes ``+0.0``, a subnormal
    ``out0`` where ``dW`` is zero is kept bit for bit; each route repeats its bits over
    ``FUSED_REPEATS`` launches from fresh copies of ``out0``."""
    out_dtype = torch.float32
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    arch, sms = _arch(device), cb.sm_count(device)
    if not _force_form(monkeypatch, form, arch):
        pytest.skip(f"no {form} weight gradient on {arch}")
    assert _promised(arch, form, out_dtype), (form, arch)
    if case in ("unsplit", "split_tail"):
        found = _find_case(arch, sms, form, case == "split_tail")
        assert found is not None, (form, case)
        sizes, n, k = found
    else:
        sizes = {
            "empty_last": EMPTY_TAIL_SIZES[0],
            "empty_split": EMPTY_TAIL_SIZES[1],
            "e32": E32_SIZES,
        }[case]
        shape = _shape_of(arch, sms, form, sizes)
        assert shape is not None, (form, case)
        n, k = shape
    sizes = list(sizes)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=71)
    _zero_column(x)
    ovw = prepare_wgrad(g, x, offs, out_dtype=out_dtype)
    dw = ovw.launch().clone()
    assert _device_route(device, ovw.plan, form, out_dtype) == "fused"
    gen = torch.Generator(device=device).manual_seed(72)
    out0 = _planted_out0(dw, sizes, gen)
    assert not bool(torch.isnan(out0).any())
    oracle = out0.clone().add_(dw)
    out = out0.clone()
    fused = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    split = fused.plan["tail_splits"] > 1
    if case == "split_tail":
        assert split
    if case == "unsplit":
        assert not split
    assert fused.plan["accumulate"] == "fused", fused.plan
    assert fused.stages == ((form + ACC, TAIL_REDUCE_ACC) if split else (form + ACC,))
    with monkeypatch.context() as mp:
        _force_composed(mp)
        composed = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert (composed.plan["accumulate"], composed.plan["accumulate_reason"]) == (
        "composed",
        "min_cuda",
    )
    assert composed.stages == ovw.stages
    assert _plan_without(fused.plan, "accumulate") == _plan_without(
        composed.plan, "accumulate", "accumulate_reason"
    )
    runs = {}
    for label, launch in (("fused", fused), ("composed", composed)):
        first = None
        for repeat in range(FUSED_REPEATS):
            out.copy_(out0)
            launch.launch()
            torch.cuda.synchronize()
            if first is None:
                first = _bits(out).clone()
            else:
                differ = int((_bits(out) != first).sum())
                assert differ == 0, (
                    f"{label}: {differ} elements differ at repeat {repeat}"
                )
        runs[label] = first
    differ = int((runs["fused"] != runs["composed"]).sum())
    assert differ == 0, f"fused differs from composed in {differ} elements"
    assert torch.equal(runs["fused"], _bits(oracle)), (
        "fused differs from out0.add_(dW32)"
    )
    result = runs["fused"].view(torch.float32)
    rows = torch.tensor([r > 0 for r in sizes], device=device)
    assert torch.equal(_bits(result[rows][:, :, 1]), _bits(out0[rows][:, :, 1])), (
        "a subnormal out0 where dW is zero was not kept"
    )
    for group, count in enumerate(sizes):
        if not count:
            want = torch.zeros_like(out0[group])
            want[0, 1], want[1, 0] = FP32_SUBNORMAL, -FP32_SUBNORMAL
            assert torch.equal(_bits(result[group]), _bits(want)), group
    print(
        f"FUSEDEQ form={form} case={case} arch={arch} E={len(sizes)} n={n} k={k} "
        f"splits={fused.plan['tail_splits']} num_tail={fused.plan['num_tail']} "
        f"stages={','.join(fused.stages)} repeats={FUSED_REPEATS}"
    )


@pytest.mark.parametrize("label,sizes", OWNER_ROWS[:1] + OWNER_ROWS[-1:])
@pytest.mark.parametrize("proj,n,k", base.BENCH_PROJ)
def test_accumulate_on_the_owner_rows(proj, n, k, label, sizes, monkeypatch):
    """The owner rows as planned (E4 and E32, both projections): an fp32 ``main_grad``
    gets bitwise ``main_grad + dW32`` (the deterministic fp32 ``dW``) on the fused route,
    the same bits at a second launch; on the E4 rows a bf16 ``main_grad`` gets
    ``main_grad + dW_bf16`` in bf16 on the composed route."""
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=33)
    dtypes = (torch.float32, torch.bfloat16) if len(sizes) == 4 else (torch.float32,)
    for out_dtype in dtypes:
        dw = _overwrite(g, x, offs, out_dtype)
        gen = torch.Generator(device=device).manual_seed(34)
        out0 = torch.randn(dw.shape, generator=gen, device=device).to(out_dtype)
        oracle = out0.clone().add_(dw)
        del dw
        out = out0.clone()
        launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
        twin = launch.stages[0].removesuffix(ACC)
        route = _device_route(device, launch.plan, twin, out_dtype)
        assert launch.plan["accumulate"] == route, launch.plan
        assert route == ("fused" if out_dtype == torch.float32 else "composed")
        launch.launch()
        torch.cuda.synchronize()
        _assert_bitwise(out, oracle, f"{label} {proj} {out_dtype} {launch.plan}")
        out.copy_(out0)
        launch.launch()
        torch.cuda.synchronize()
        _assert_bitwise(out, oracle, f"{label} {proj} {out_dtype} second launch")
        del launch, out, out0, oracle
        torch.cuda.empty_cache()


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_no_rows_launch_nothing_and_leave_out(out_dtype):
    """``sum_m == 0``: the accumulate launch has no kernel and no add, and ``out`` keeps
    its bits (NaN padding included)."""
    device = base._require_program("wgrad", out_dtype)
    n, k = 256, 512
    x, _w, g, offs = base._inputs([0, 0], n, k, device)
    buffer = torch.empty(2, n + 2, k + 32, dtype=out_dtype, device=device)
    _bits(buffer).fill_(0x7FC0 if out_dtype == torch.bfloat16 else 0x7FC00000)
    out = buffer[:, :n, :k]
    before = _bits(buffer).clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.launches == () and launch.add_from is None
    assert launch.launch() is out
    torch.cuda.synchronize()
    assert torch.equal(_bits(buffer), before)


@pytest.mark.parametrize("layout", ["row_padded", "one_group", "m_indptr", "inference"])
def test_accumulate_layouts(layout, monkeypatch):
    """A row- and expert-padded ``out`` (the padding keeps its bits), one group, the
    ``m_indptr`` form (``num_groups=E``) and an inference tensor: each gets the bits of
    ``out0.add_(dW)``."""
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = (
        ([9000], 512, 1024) if layout == "one_group" else (base.RAGGED_E6, 512, 512)
    )
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=36)
    e = len(sizes)
    dw = _overwrite(g, x, offs, torch.float32)
    gen = torch.Generator(device=device).manual_seed(37)
    out0 = torch.randn(e, n, k, generator=gen, device=device)
    oracle = out0.clone().add_(dw)
    if layout == "row_padded":
        buffer = torch.full((e, n + 2, k + 32), float("nan"), device=device)
        out = buffer[:, :n, :k]
        out.copy_(out0)
        before = buffer.clone()
        prepare_wgrad(g, x, offs, out=out, accumulate=True).launch()
        torch.cuda.synchronize()
        _assert_bitwise(out, oracle, layout)
        pad = torch.ones_like(buffer, dtype=torch.bool)
        pad[:, :n, :k] = False
        assert bool(torch.isnan(buffer[pad]).all()) and bool(
            torch.isnan(before[pad]).all()
        )
        return
    if layout == "inference":
        with torch.inference_mode():
            out = out0.clone()
        assert out.is_inference()
    else:
        out = out0.clone()
    if layout == "m_indptr":
        indptr = base._m_indptr(sizes, device)
        launch = prepare_wgrad(g, x, indptr, out=out, num_groups=e, accumulate=True)
    else:
        launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    launch.launch()
    torch.cuda.synchronize()
    _assert_bitwise(out, oracle, layout)


@pytest.mark.parametrize("route", ["fused", "composed"])
def test_accumulate_on_two_concurrent_streams(route, monkeypatch):
    """Two prepared accumulates of a split-tail shape into separate outputs (on the
    composed route each owning its scratch), launched interleaved on two streams while
    copies and GEMMs run on a third: every result has the oracle's bits on either
    route."""
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _route_case(device, route, True, monkeypatch)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=38)
    dw = _overwrite(g, x, offs, torch.float32)
    out0 = torch.randn(dw.shape, device=device)
    oracle = out0.clone().add_(dw)
    launches = [
        prepare_wgrad(g, x, offs, out=out0.clone(), accumulate=True) for _ in range(2)
    ]
    assert [launch.plan["accumulate"] for launch in launches] == [route] * 2
    if route == "composed":
        assert launches[0].add_from.data_ptr() != launches[1].add_from.data_ptr()
    else:
        assert launches[0].add_from is None and launches[1].add_from is None
    streams = [torch.cuda.Stream(device) for _ in range(3)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    src = torch.randn(1 << 24, device=device)
    dst = torch.empty_like(src)
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    results = [[], []]
    for _ in range(6):
        for index in range(2):
            with torch.cuda.stream(streams[index]):
                launches[index].out.copy_(out0)
                results[index].append(launches[index].launch().clone())
        with torch.cuda.stream(streams[2]):
            dst.copy_(src)
            torch.mm(a, a, out=c)
    for stream in streams:
        main.wait_stream(stream)
    torch.cuda.synchronize()
    for index in range(2):
        for out in results[index]:
            _assert_bitwise(out, oracle, f"stream {index}")


@pytest.mark.parametrize("route", ["fused", "composed"])
@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "split_tail"])
def test_graph_replay_accumulates_at_every_replay(split, route, monkeypatch):
    """A captured accumulate launch adds at every replay on either route: two replays
    without a reset give bitwise ``out0.add_(dW).add_(dW)``, also after new group
    boundaries (the same ``sum_m``) are written into ``offs``; the composed route's add is
    captured with the programs."""
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _route_case(device, route, split, monkeypatch)
    second = [sizes[3], sizes[1], sizes[2], sizes[0]]
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=39)
    out0 = torch.randn(len(sizes), n, k, device=device)
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.plan["accumulate"] == route
    ovw = prepare_wgrad(g, x, offs, out_dtype=torch.float32)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()  # the first launch on a stream outside the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    for layout in (sizes, second, sizes):
        offs.copy_(base._offs(layout, device))
        dw = ovw.launch().clone()
        out.copy_(out0)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(out, out0.clone().add_(dw).add_(dw), f"layout {layout}")


@pytest.mark.parametrize("route", ["fused", "composed"])
def test_prepared_accumulate_allocates_only_at_preparation(route, monkeypatch):
    """The composed preparation allocates the scratch (at least ``E * N * K`` fp32
    elements); the fused one allocates what the overwrite preparation of the same call
    does (the descriptor slots and the tail partials: no scratch); after the first launch
    on a stream neither route's launch allocates device memory."""
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _route_case(device, route, True, monkeypatch)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=40)
    out = torch.zeros(len(sizes), n, k, device=device)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.plan["accumulate"] == route
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated(device) - before
    if route == "composed":
        assert grown >= launch.add_from.numel() * 4
    else:
        assert launch.add_from is None
        before = torch.cuda.memory_allocated(device)
        ovw = prepare_wgrad(g, x, offs, out=out)
        torch.cuda.synchronize()
        assert grown == torch.cuda.memory_allocated(device) - before, grown
        del ovw
    launch.launch()
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    for _ in range(3):
        launch.launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == stats


@pytest.mark.parametrize("route", ["fused", "composed"])
def test_accumulate_bumps_the_version_counter(route, monkeypatch):
    """Every accumulate launch bumps ``out._version`` once (the fused route through
    ``torch.autograd.graph.increment_version``, the composed one through ``out.add_``); an
    inference tensor is accumulated into without an error, with the oracle's bits."""
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _route_case(device, route, False, monkeypatch)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=41)
    out = torch.zeros(len(sizes), n, k, device=device)
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.plan["accumulate"] == route
    version = out._version
    launch.launch()
    launch.launch()
    assert out._version == version + 2
    with torch.inference_mode():
        frozen = torch.zeros(len(sizes), n, k, device=device)
    assert frozen.is_inference()
    prepare_wgrad(g, x, offs, out=frozen, accumulate=True).launch()
    torch.cuda.synchronize()
    _assert_bitwise(
        frozen, torch.zeros_like(out).add_(_overwrite(g, x, offs, torch.float32))
    )


@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "split_tail"])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("form", list(FORMS))
def test_torch_deterministic_mode_changes_nothing(form, out_dtype, split, monkeypatch):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``; with
    ``torch.utils.deterministic.fill_uninitialized_memory`` on, its default, so torch fills
    each fresh scratch with NaN) an accumulate of every form, with and without the split
    tail, prepared or one-shot, raises no alert, warns nothing and gives the bits it gives
    with the mode off."""
    assert torch.utils.deterministic.fill_uninitialized_memory
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _device_case(device, form, split, monkeypatch, out_dtype)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=42)
    out0 = torch.randn(len(sizes), n, k, device=device).to(out_dtype)
    want = out0.clone()
    pkg.cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out=want, accumulate=True)
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        for warn_only in (False, True):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                out = out0.clone()
                prepare_wgrad(g, x, offs, out=out, accumulate=True).launch()
                one_shot = out0.clone()
                pkg.cake_moe_grouped_gemm_det_train_wgrad(
                    g, x, offs, out=one_shot, accumulate=True
                )
            torch.cuda.synchronize()
            assert [w for w in rec if w.category is UserWarning] == [], warn_only
            assert torch.equal(_bits(out), _bits(want))
            assert torch.equal(_bits(one_shot), _bits(want))
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


# ---------------------------------------------------------------------------
# Autograd (GPU)
# ---------------------------------------------------------------------------


def _autograd_case(monkeypatch, out_dtype=torch.float32, split=False):
    """``(device, sizes, x0, w0, g1, g2, offs)`` of a ``main_halves`` shape with or without
    a split tail on the device, with every program of the wrapper available."""
    base._require_program("fwd")
    base._require_program("dgrad")
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    sizes, n, k = _device_case(device, "main_halves", split, monkeypatch, out_dtype)
    x0, w0, g1, offs = base._inputs(sizes, n, k, device, seed=51)
    g2 = base._inputs(sizes, n, k, device, seed=52)[2]
    return device, sizes, x0, w0, g1, g2, offs


def _composed_main_grad(mg0, grads, x, offs):
    """``mg0`` plus each deterministic weight gradient of ``grads`` added in ``mg0``'s dtype
    (AccumulateGrad of the deterministic ``dW`` in that dtype)."""
    mg = mg0.clone()
    for g in grads:
        mg.add_(
            pkg.cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out_dtype=mg0.dtype)
        )
    return mg


def _fused_gemm(offs):
    return lambda a, b: pkg.cake_moe_grouped_gemm_det(
        a, b, offs, fuse_wgrad_accumulation=True
    )


def _run_micro_steps(fn, x0, w0, mg0, grads, ckpt, early_stop, attribute=None):
    """Micro-steps of ``fn(x, w)`` with ``w.main_grad = mg0.clone()``; returns ``(w, x
    grads per step)``."""
    w = w0.clone().requires_grad_(True)
    w.main_grad = mg0.clone()
    for key, value in (attribute or {}).items():
        setattr(w, key, value)
    dxs = []
    for g in grads:
        x = x0.clone().requires_grad_(True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            if ckpt is None:
                y = fn(x, w)
            elif ckpt == "nested":

                def inner(a, b):
                    return torch.utils.checkpoint.checkpoint(
                        fn, a, b, use_reentrant=False
                    )

                y = torch.utils.checkpoint.checkpoint(inner, x, w, use_reentrant=False)
            else:
                y = torch.utils.checkpoint.checkpoint(fn, x, w, use_reentrant=False)
            y.backward(g)
        dxs.append(x.grad)
    torch.cuda.synchronize()
    return w, dxs


@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
@pytest.mark.parametrize("ckpt", [None, "non_reentrant", "nested"])
@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "split_tail"])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_autograd_accumulates_into_main_grad(
    out_dtype, split, ckpt, early_stop, monkeypatch
):
    """Two micro-steps with ``fuse_wgrad_accumulation=True`` (plain, non-reentrant and
    nested checkpoints, early stop on and off; with and without a split tail) leave
    bitwise ``main_grad == mg0 + dW1 + dW2`` (each deterministic ``dW`` in ``main_grad``'s
    dtype added in that dtype), the same bits in a second run, within the budget of the
    fp64 sum; ``x.grad`` is bitwise the default mode's and ``w.grad`` stays None."""
    device, sizes, x0, w0, g1, g2, offs = _autograd_case(monkeypatch, out_dtype, split)
    gen = torch.Generator(device=device).manual_seed(53)
    mg0 = torch.randn(w0.shape, generator=gen, device=device).to(out_dtype)
    w, dxs = _run_micro_steps(
        _fused_gemm(offs), x0, w0, mg0, (g1, g2), ckpt, early_stop
    )
    assert w.grad is None
    want = _composed_main_grad(mg0, (g1, g2), x0, offs)
    _assert_bitwise(w.main_grad, want, "main_grad")
    again, _ = _run_micro_steps(
        _fused_gemm(offs), x0, w0, mg0, (g1, g2), ckpt, early_stop
    )
    assert torch.equal(_bits(again.main_grad), _bits(w.main_grad)), "run to run"
    dws = [base._reference("wgrad", g, x0, sizes).double() for g in (g1, g2)]
    _assert_within_budget(w.main_grad, _steps_expected(mg0, dws))
    _, default_dxs = _run_micro_steps(
        lambda a, b: pkg.cake_moe_grouped_gemm_det(a, b, offs),
        x0,
        w0,
        mg0,
        (g1, g2),
        ckpt,
        early_stop,
    )
    for got, expect in zip(dxs, default_dxs, strict=True):
        assert torch.equal(got, expect)


def test_autograd_bf16_main_grad_is_accumulate_grad(monkeypatch):
    """A bf16 ``main_grad`` gets the bits AccumulateGrad gives: the default wrapper's bf16
    ``w.grad`` accumulated by autograd over two micro-steps from ``w.grad = mg0`` equals
    ``w.main_grad`` of the fused wrapper from ``mg0``."""
    device, _sizes, x0, w0, g1, g2, offs = _autograd_case(monkeypatch, torch.bfloat16)
    mg0 = torch.randn(w0.shape, device=device).to(torch.bfloat16)
    w, _ = _run_micro_steps(_fused_gemm(offs), x0, w0, mg0, (g1, g2), None, True)
    plain = w0.clone().requires_grad_(True)
    plain.grad = mg0.clone()
    for g in (g1, g2):
        x = x0.clone().requires_grad_(True)
        pkg.cake_moe_grouped_gemm_det(x, plain, offs).backward(g)
    torch.cuda.synchronize()
    _assert_bitwise(w.main_grad, plain.grad, "AccumulateGrad")


@pytest.mark.parametrize("zero_out", [False, True], ids=["empty", "zeros"])
def test_autograd_returns_megatron_placeholder(zero_out, monkeypatch):
    """With ``w.grad_added_to_main_grad``: the backward sets it, adds into ``main_grad``
    and returns a fresh gradient of ``w``'s shape and dtype (zeros under
    ``zero_out_wgrad``), which AccumulateGrad stores as ``w.grad``."""
    device, _sizes, x0, w0, g1, _g2, offs = _autograd_case(monkeypatch)
    mg0 = torch.zeros(w0.shape, device=device)
    attribute = {"grad_added_to_main_grad": False, "zero_out_wgrad": zero_out}
    w, _ = _run_micro_steps(
        _fused_gemm(offs), x0, w0, mg0, (g1,), None, True, attribute
    )
    assert w.grad_added_to_main_grad is True
    assert w.grad is not None and w.grad.shape == w0.shape and w.grad.dtype == w0.dtype
    if zero_out:
        assert int(w.grad.count_nonzero()) == 0
    _assert_bitwise(w.main_grad, _composed_main_grad(mg0, (g1,), x0, offs), "main_grad")


@pytest.mark.parametrize("split", [False, True], ids=["unsplit", "split_tail"])
def test_graph_captures_the_fused_backward(split, monkeypatch):
    """The forward and the fused backward of ``cake_moe_grouped_gemm_det(...,
    fuse_wgrad_accumulation=True)`` capture into one CUDA graph (the backward binds the
    context without querying the captured stream, and prepares its accumulate launch: on
    the fused route in place, on the composed route with the scratch from the graph's
    pool): ``r`` replays from ``mg0`` give bitwise ``mg0``
    plus ``r`` adds of the deterministic fp32 ``dW`` (AccumulateGrad's bits), and
    ``x.grad`` holds the eager backward's bits."""
    device, _sizes, x0, w0, g1, _g2, offs = _autograd_case(
        monkeypatch, torch.float32, split
    )
    gen = torch.Generator(device=device).manual_seed(57)
    mg0 = torch.randn(w0.shape, generator=gen, device=device)
    w = w0.clone().requires_grad_(True)
    w.main_grad = mg0.clone()
    x = x0.clone().requires_grad_(True)
    fn = _fused_gemm(offs)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        fn(x, w).backward(g1)  # every program loaded outside the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    eager_dx = x.grad.clone()
    x.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(x, w).backward(g1)
    torch.cuda.synchronize()
    assert w.grad is None
    dw = pkg.cake_moe_grouped_gemm_det_train_wgrad(
        g1, x0, offs, out_dtype=torch.float32
    )
    for replays in (1, 3):
        w.main_grad.copy_(mg0)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        want = mg0.clone()
        for _ in range(replays):
            want.add_(dw)
        _assert_bitwise(w.main_grad, want, f"{replays} replays")
        assert torch.equal(x.grad, eager_dx)


def test_autograd_flag_errors_and_side_effects(monkeypatch):
    """With the generated programs on a split-tail shape: a missing ``main_grad`` at the
    forward and a wrong one at the backward raise ``ValueError``; ``w`` as a reentrant
    checkpoint's input arrives without ``main_grad`` (``ValueError``); ``retain_graph``
    twice adds twice; a frozen ``w`` leaves ``main_grad``; ``create_graph`` raises
    ``NotImplementedError``; a ``torch.autograd.grad`` whose ``inputs`` exclude ``w``
    still adds; a row-padded ``main_grad`` and the ``m_indptr`` form get the same bits;
    under torch's deterministic mode (strict) the backward adds the same bits without a
    warning."""
    device, sizes, x0, w0, g1, _g2, offs = _autograd_case(monkeypatch, split=True)
    gemm = _fused_gemm(offs)
    mg0 = torch.randn(w0.shape, device=device)
    dw1 = pkg.cake_moe_grouped_gemm_det_train_wgrad(
        g1, x0, offs, out_dtype=torch.float32
    )
    once = mg0.clone().add_(dw1)

    def leaves(main_grad=True, requires_grad=True):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(requires_grad)
        if main_grad:
            w.main_grad = mg0.clone()
        return x, w

    x, w = leaves(main_grad=False)
    with pytest.raises(ValueError, match="main_grad"):
        gemm(x, w)
    x, w = leaves()
    y = gemm(x, w)
    w.main_grad = torch.zeros(1, device=device)
    with pytest.raises(ValueError, match="w.main_grad must be"):
        y.backward(g1)
    x, w = leaves()
    y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=True)
    with pytest.raises(ValueError, match="main_grad"):
        y.backward(g1)
    x, w = leaves()
    y = gemm(x, w)
    y.backward(g1, retain_graph=True)
    y.backward(g1)
    torch.cuda.synchronize()
    _assert_bitwise(w.main_grad, mg0.clone().add_(dw1).add_(dw1), "retain_graph")
    x, w = leaves(requires_grad=False)
    gemm(x, w).backward(g1)
    torch.cuda.synchronize()
    assert torch.equal(w.main_grad, mg0) and x.grad is not None
    x, w = leaves()
    with pytest.raises(NotImplementedError, match="create_graph"):
        torch.autograd.grad(gemm(x, w), (x, w), g1, create_graph=True)
    x, w = leaves()
    torch.autograd.grad(gemm(x, w), (x,), g1)
    torch.cuda.synchronize()
    _assert_bitwise(w.main_grad, once, "autograd.grad(inputs=(x,))")
    x, w = leaves()
    buffer = torch.full(
        (w0.shape[0], w0.shape[1], w0.shape[2] + 32), float("nan"), device=device
    )
    w.main_grad = buffer[:, :, : w0.shape[2]]
    w.main_grad.copy_(mg0)
    gemm(x, w).backward(g1)
    torch.cuda.synchronize()
    _assert_bitwise(w.main_grad, once, "row-padded main_grad")
    assert bool(torch.isnan(buffer[:, :, w0.shape[2] :]).all())
    x, w = leaves()
    indptr = base._m_indptr(sizes, device)
    pkg.cake_moe_grouped_gemm_det(x, w, indptr, fuse_wgrad_accumulation=True).backward(
        g1
    )
    torch.cuda.synchronize()
    _assert_bitwise(w.main_grad, once, "m_indptr")
    x, w = leaves()
    y = gemm(x, w)
    with base._torch_deterministic(), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        y.backward(g1)
    torch.cuda.synchronize()
    assert [r for r in rec if r.category is UserWarning] == []
    _assert_bitwise(w.main_grad, once, "torch deterministic mode")


@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_autograd_fuse_checkpoint_early_stop_skips_the_forward_launch(
    early_stop, monkeypatch
):
    """With ``fuse_wgrad_accumulation=True`` the wrapper still launches its forward after
    autograd saved the inputs: a non-reentrant checkpoint launches the forward once with
    early stop on and twice with it off, with the bits of the plain call for ``Y``,
    ``dX`` and ``main_grad``."""
    device, _sizes, x0, w0, g1, _g2, offs = _autograd_case(monkeypatch)
    mg0 = torch.zeros(w0.shape, device=device)
    launches = collections.Counter()
    fwd_op = cake_autograd._fwd_op

    def counted(*args):
        launches["fwd"] += 1
        fwd_op(*args)

    monkeypatch.setattr(cake_autograd, "_fwd_op", counted)
    gemm = _fused_gemm(offs)

    def run(checkpointed):
        launches.clear()
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        w.main_grad = mg0.clone()
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            if checkpointed:
                y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
            else:
                y = gemm(x, w)
            y.backward(g1)
        torch.cuda.synchronize()
        return (y.detach(), x.grad, w.main_grad), launches["fwd"]

    plain, n0 = run(False)
    checkpointed, n1 = run(True)
    assert n0 == 1 and n1 == (1 if early_stop else 2)
    for a, b in zip(checkpointed, plain, strict=True):
        assert torch.equal(a, b)
