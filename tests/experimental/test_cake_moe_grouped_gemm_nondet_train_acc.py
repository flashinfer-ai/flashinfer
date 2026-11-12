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

Weight-gradient accumulation of the non-deterministic ragged BF16 MoE grouped GEMM
package: ``cake_moe_grouped_gemm_nondet_train_prepare_wgrad`` / ``cake_moe_grouped_gemm_nondet_train_wgrad`` with
``accumulate=True`` (``out[e] += G[e].T @ X[e]``) and ``cake_moe_grouped_gemm_nondet`` with
``fuse_wgrad_accumulation=True`` (``w.main_grad += dW`` in the backward).

On the CPU (no program is loaded): the registry's accumulate stages (``<stage>_acc``)
against the exact set this release ships (``SHIPPED_ACC_STAGES``, which the README and
rst feature matrices state too) and against the stages they are builds of (their twins),
the ``build.json`` entries, the accumulate plans against the overwrite plans over the
owner rows at every SM count, the fused / composed route choice (the 256-wide k tile, a
two-launch tail-reduce tail, a stage without an accumulate build, a build that cannot
load by the loader's static rule, the alignment; the ordered and the unordered
one-launch tail both fuse), the refusal of an architecture listed in both tail lists or
in neither where no ``tail_reduce`` stage serves it, the checks of ``out``, the dormant
alert (only before an order-varying launch, whatever the route: a split tail planned
under ``unordered_tail_archs``, which the shipped constants leave empty, so no shipped
plan varies), the signatures, the facade's handoff (no deterministic route) and the
autograd flag's refusals.  On a GPU: the fused route against its bitwise oracles
(``out0.add_(dW32)`` on the groups with rows, split ordered tails included) and the
fp64 reference, on forced forms and on the benchmark rows as planned, bitwise run to run
with the stream's tail slots garbage-filled before every launch, the composed route
against ``out.add_``, empty groups, padding, ``sum_m == 0``, short groups, concurrent
streams and graph replay (bitwise an isolated eager launch), allocations, the version
counter, the autograd flag, and a fused split-tail backward under torch's deterministic
mode (no alert, the bits of the mode-off run).  A fused case the shipped set promises
fails unless the call is fused; it skips (``"cannot load"`` in the reason) only where the
build's library cannot load in this environment (neither prebuilt nor a JIT toolkit at
its ``min_cuda``), where the call composes instead.

``python -m tests.experimental.test_cake_moe_grouped_gemm_nondet_train_acc sanitize`` (from the
repository root) runs one small launch of every accumulate build the device serves, with
an unsplit and a split tail, for ``compute-sanitizer --tool memcheck | racecheck |
synccheck``.
"""

import contextlib
import importlib.util
import inspect
import os
import re
import shutil
import sys
import types
import warnings

import pytest
import torch
import torch.utils.checkpoint

from tests.experimental import test_cake_moe_grouped_gemm_nondet_train as base

pkg = base.pkg
cb = base.cb
cake_jit = base.cake_jit
cake_autograd = base.cake_autograd
ROUTES = base.ROUTES
PROGRAMS = base.PROGRAMS
HOST_PLAN_CONSTANTS = base.HOST_PLAN_CONSTANTS
DTYPE_NAMES = base.DTYPE_NAMES
ACC = cb.ACC_SUFFIX
# The stages with an accumulate build: the planner-reachable 512-wide k tile forms and
# the queue walk (``cake_backend.wgrad_queue_walk``).
TWIN_FORMS = ("main_halves", "main_wave_halves", "main_halves_k256t", cb.QUEUE_STAGE)
# The accumulate builds this release ships, route -> architecture -> stages: fp32 ``out``
# only (a bf16 ``out`` composes: the bf16 overwrite and a same-dtype ``add_``).  The
# registry serves exactly these (``test_acc_bindings_are_the_shipped_set``), and the README
# and rst feature matrices state the same (``test_docs_list_the_shipped_acc_builds``).
SHIPPED_ACC_STAGES = {
    "cake_moe_grouped_gemm_wgrad_f32_k512": {
        "sm_100a": ("main_halves_acc", "main_wave_halves_acc", "main_halves_queue_acc"),
        "sm_103a": ("main_halves_acc", "main_wave_halves_acc", "main_halves_queue_acc"),
        "sm_107a": ("main_halves_acc", "main_halves_k256t_acc"),
    },
}
# The output dtypes with accumulate builds (the fused cases run these).
FUSED_DTYPES = tuple(
    dtype
    for dtype, name in DTYPE_NAMES.items()
    if cake_jit.record_name("wgrad", out_dtype=name, tile_k=512) in SHIPPED_ACC_STAGES
)
# The SM counts the CPU plan sweep runs at (every architecture at every count).
SWEEP_SM_COUNTS = (148, 152, 160, 208, 212, 216)
# The owner rows of the sweep: the #5678 benchmark rows plus one group of all rows.
OWNER_ROWS = base.BENCH_WGRAD_ROWS + (("e1_m131072", [131072]),)
# Forced 512-wide k tile shapes of the GPU cases: an empty and a one-row group.
ACC_SIZES = [4100, 0, 4170, 1]
# Empty groups placed where the order puts them in the tail (planned walk engaged, split
# tail): the kernel tests' lists.
EMPTY_TAIL_SIZES = (
    [640, 384, 768, 448, 512, 576, 0, 0],
    [4099] * 4 + [0, 0] + [130] * 2,
)
INT_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
FP32_SUBNORMAL = 1e-40
prepare_wgrad = base.prepare_wgrad


def _plan_without(plan, *keys):
    return {key: value for key, value in plan.items() if key not in keys}


def _acc_bindings():
    """``(route, stage, arch)`` of every accumulate stage the registry serves."""
    found = []
    for name, route in ROUTES.items():
        for item in route["stages"]:
            if not item["name"].endswith(ACC):
                continue
            for arch in route["arches"]:
                if arch in PROGRAMS[item["program"]]["arches"]:
                    found.append((name, item["name"], arch))
    return found


def _shipped_bindings():
    """``(route, stage, arch)`` of every accumulate build of ``SHIPPED_ACC_STAGES``."""
    return [
        (name, stage, arch)
        for name, arches in SHIPPED_ACC_STAGES.items()
        for arch, stages in arches.items()
        for stage in stages
    ]


def _promised(arch, form, out_dtype):
    """Whether this release ships the accumulate build of ``form`` for ``out_dtype`` on ``arch``."""
    name = cake_jit.record_name("wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=512)
    return form + ACC in SHIPPED_ACC_STAGES.get(name, {}).get(arch, ())


def _fake_device(monkeypatch, arch, sms, *, checks=True, loadable=True):
    """Run the host of ``_prepare_wgrad`` without a GPU: the operands' device stands for an
    ``arch`` device with ``sms`` SMs, programs load as no-op modules, the descriptor slots
    are plain bytes and every accumulate build's library counts as loadable (``loadable``;
    the loader's rule has its own tests).  ``checks=False`` also skips the accumulate checks
    of ``out`` (for meta tensors, which have no addresses; the checks have their own tests)."""
    monkeypatch.setattr(
        cb, "_check_device", lambda tensors: next(iter(tensors.values())).device
    )
    monkeypatch.setattr(cb, "arch_for", lambda device: arch)
    monkeypatch.setattr(cb, "sm_count", lambda device: sms)
    monkeypatch.setattr(cb, "load_module", lambda *args, **kwargs: base._NoModule())
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


def _meta_operands(sizes, n, k, out_dtype):
    sum_m, e = sum(sizes), len(sizes)
    meta = torch.device("meta")
    g = torch.empty(sum_m, n, dtype=torch.bfloat16, device=meta)
    x = torch.empty(sum_m, k, dtype=torch.bfloat16, device=meta)
    offs = torch.empty(e, dtype=torch.int32, device=meta)
    out = torch.empty(e, n, k, dtype=out_dtype, device=meta)
    return g, x, offs, out


def _cpu_operands(sizes, n, k, out_dtype):
    sum_m, e = sum(sizes), len(sizes)
    g = torch.zeros(sum_m, n, dtype=torch.bfloat16)
    x = torch.zeros(sum_m, k, dtype=torch.bfloat16)
    offs = torch.tensor(sizes, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    out = torch.zeros(e, n, k, dtype=out_dtype)
    return g, x, offs, out


def _force_k512(monkeypatch, arch):
    """The 512-wide k tile at any rows per group on ``arch``, with the planner's stage rules."""
    monkeypatch.setitem(HOST_PLAN_CONSTANTS["k512_min_rows_per_group"], arch, 1)


def _tail_lists(arch, *, ordered, unordered):
    """``ordered_tail_archs`` and ``unordered_tail_archs`` of ``HOST_PLAN_CONSTANTS``
    with ``arch`` listed in the first exactly when ``ordered`` and in the second exactly
    when ``unordered`` (the other architectures keep their entries)."""
    lists = {}
    for key, listed in (
        ("ordered_tail_archs", ordered),
        ("unordered_tail_archs", unordered),
    ):
        others = [a for a in HOST_PLAN_CONSTANTS[key] if a != arch]
        lists[key] = others + [arch] if listed else others
    return lists


def _set_tail_lists(monkeypatch, arch, *, ordered, unordered):
    """Put ``_tail_lists`` into ``HOST_PLAN_CONSTANTS`` (the preparation's constants)."""
    for key, value in _tail_lists(arch, ordered=ordered, unordered=unordered).items():
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, key, value)


def _prepare_pair(g, x, offs, out, num_groups=None):
    """The overwrite and the accumulate preparation of the same call (no launch)."""
    ovw = cb._prepare_wgrad(g, x, offs, out, None, num_groups)
    acc = cb._prepare_wgrad(g, x, offs, out, None, num_groups, accumulate=True)
    return ovw, acc


def _assert_twin_launch(ovw, acc, arch):
    """``acc`` is ``ovw`` in accumulate mode: the same route and plan apart from the
    accumulate keys, and the binding of its route; a fused launch is one launch with no
    tail-reduce stage, planned with the ordered one-launch tail (shipped constants)."""
    assert ovw.plan["accumulate"] == "off" and "accumulate_reason" not in ovw.plan
    assert acc.record_name == ovw.record_name
    assert _plan_without(acc.plan, "accumulate", "accumulate_reason") == _plan_without(
        ovw.plan, "accumulate"
    )
    tile_k = ovw.plan["tile_k"]
    stage = ovw.stages[0] if ovw.stages else None
    mode = acc.plan["accumulate"]
    assert mode in cb.ACCUMULATE_MODES and mode != "off"
    if stage is not None:
        want = cb.wgrad_accumulate_mode(
            ovw.record_name,
            stage,
            tile_k,
            arch,
            aligned=cb._accumulate_aligned(acc.out),
        )
        assert (mode, acc.plan.get("accumulate_reason")) == want, (stage, arch)
    if not ovw.stages:  # no row: nothing launches either way
        assert acc.stages == () and acc.add_from is None and not acc.zero_fill
        return
    if mode == "fused":
        # The fused accumulate binds the twin's accumulate stage, or the queue walk's where a
        # ``queue_walk`` rule names the fused accumulate (``wgrad_queue_walk``).
        queue_acc = cb.QUEUE_STAGE + ACC
        if acc.stages == (queue_acc,) and stage != cb.QUEUE_STAGE:
            e = acc.out.shape[0]
            assert stage in cb.QUEUE_SELECTED_STAGES, (stage, arch)
            assert cb.wgrad_queue_walk(
                stage, "accumulate", arch, cb.sm_count(acc.out.device), e
            ), (stage, arch, e)
            stage = cb.QUEUE_STAGE
        elif stage in cb.QUEUE_SELECTED_STAGES:
            assert not (
                cb.wgrad_queue_walk(
                    stage,
                    "accumulate",
                    arch,
                    cb.sm_count(acc.out.device),
                    acc.out.shape[0],
                )
                and cake_jit.stage_served(acc.record_name, queue_acc, arch)
            ), (stage, arch, acc.stages)
        assert acc.stages == (stage + ACC,) and len(acc.launches) == 1
        assert acc.plan["tail_reduce_grid"] is None and acc.plan["launches"] == 1
        assert acc.plan["ordered_tail"] and not acc.plan["unordered_tail"], acc.plan
        assert acc.add_from is None and acc.bump_version
    else:
        assert acc.stages == ovw.stages
        assert acc.add_from is not None and not acc.bump_version
        assert (
            acc.add_from.shape == acc.out.shape and acc.add_from.dtype == acc.out.dtype
        )
        assert acc.add_from.is_contiguous()
    assert not acc.zero_fill


# ---------------------------------------------------------------------------
# B. The registry and the host (CPU)
# ---------------------------------------------------------------------------


def test_acc_bindings_are_the_shipped_set():
    """The registry serves exactly the accumulate builds of ``SHIPPED_ACC_STAGES``: the
    fp32 512-wide k tile route serves ``main_halves_acc`` on sm_100a / sm_103a / sm_107a,
    ``main_wave_halves_acc`` and ``main_halves_queue_acc`` on sm_100a / sm_103a and
    ``main_halves_k256t_acc`` on sm_107a;
    the bf16 routes serve none (a bf16 ``out`` composes).  A render that lost or added an
    accumulate stage fails here."""
    assert sorted(_acc_bindings()) == sorted(_shipped_bindings())
    for arch in cake_jit.ARCHES:
        for name in DTYPE_NAMES.values():
            route = cake_jit.record_name("wgrad", out_dtype=name, tile_k=512)
            want = bool(SHIPPED_ACC_STAGES.get(route, {}).get(arch))
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
    the twin forms of ``SHIPPED_ACC_STAGES`` for an fp32 ``out`` only (no bf16), and the
    rst feature row says fp32 and lists the same forms by their short names."""
    (route,) = SHIPPED_ACC_STAGES
    readme = os.path.join(
        _REPO,
        "flashinfer",
        "experimental",
        "cake",
        "cake_moe_grouped_gemm_nondet_train",
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
        assert forms == set(SHIPPED_ACC_STAGES[route][arch]), (arch, cell)
    rst = os.path.join(_REPO, "docs", "api", "cake_moe_grouped_gemm_nondet_train.rst")
    line = _doc_matrix_row(rst, "``wgrad`` accumulate, fused")
    label, rest = line.split("``out``", 1)
    assert "fp32" in label and "bf16" not in line, line
    short = {
        "main_halves_acc": "planned",
        "main_wave_halves_acc": "wave",
        "main_halves_k256t_acc": "k256t",
        "main_halves_queue_acc": "queue",
    }
    cells = re.split(r"\s{2,}", rest.strip())
    assert len(cells) == len(cake_jit.ARCHES), line
    for arch, cell in zip(cake_jit.ARCHES, cells, strict=True):
        want = sorted(short[stage] for stage in SHIPPED_ACC_STAGES[route][arch])
        assert sorted(part.strip() for part in cell.split(",")) == want, (arch, line)


def test_acc_stages_are_builds_of_their_twins():
    """Every accumulate stage the registry serves is ``<twin>_acc`` on a 512-wide k tile
    weight-gradient route, for a twin among the planner-reachable halves forms that the
    route serves on the same architecture; its program is another build with the twin
    program's argument plan, launch shape and FFI entry, under the twin's
    specializations; ``main_halves_k256t_acc`` is served only where ``main_halves_acc``
    is; no new name contains ``det``."""
    bindings = _acc_bindings()
    assert bindings, "the registry serves no accumulate build"
    for name, stage, arch in bindings:
        route = ROUTES[name]
        assert route["op"] == "wgrad" and name.endswith("_k512"), name
        twin = stage[: -len(ACC)]
        assert twin in TWIN_FORMS, (name, stage)
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
        assert acc_program["ffi_entry"] == twin_program["ffi_entry"]
        for label in (name, stage, acc_binding["program"], acc_program["library"]):
            assert "det" not in label.lower(), label
        if twin == "main_halves_k256t":
            assert cake_jit.stage_served(name, "main_halves" + ACC, arch), (name, arch)
        dtype = (
            "float32"
            if name.startswith("cake_moe_grouped_gemm_wgrad_f32")
            else "bfloat16"
        )
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


@pytest.mark.parametrize("sm_count", SWEEP_SM_COUNTS)
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_accumulate_plans_equal_the_overwrite_plans(arch, sm_count, monkeypatch):
    """Over every owner row (gate/up and down, E in {1, 4, 8, 16, 32}) and both output
    dtypes, at every SM count: the accumulate preparation selects the overwrite's route,
    tile, stage and plan (every plan key but ``accumulate`` / ``accumulate_reason``
    equal), binds ``<stage>_acc`` exactly where the shipped set has that build (an fp32
    ``out``: owner rows take a halves stage, so every fp32 owner row is fused, with no
    tail-reduce stage), and else composes (a bf16 ``out``: ``"stage"``; the twin's stages
    into a scratch of out's dtype).  The overwrite plans of the two output dtypes are
    equal."""
    _fake_device(monkeypatch, arch, sm_count, checks=False)
    for label, sizes in OWNER_ROWS:
        for proj, (n, k) in base.BENCH_PROJ.items():
            plans = {}
            for out_dtype in (torch.bfloat16, torch.float32):
                g, x, offs, out = _meta_operands(sizes, n, k, out_dtype)
                ovw, acc = _prepare_pair(g, x, offs, out)
                _assert_twin_launch(ovw, acc, arch)
                assert ovw.plan["tile_k"] == 512, (label, proj, ovw.plan)
                assert "halves" in ovw.stages[0], (label, proj, ovw.stages)
                if _promised(arch, ovw.stages[0], out_dtype):
                    assert acc.plan["accumulate"] == "fused", (label, proj, acc.plan)
                else:
                    assert (acc.plan["accumulate"], acc.plan["accumulate_reason"]) == (
                        "composed",
                        "stage",
                    ), (label, proj, out_dtype)
                plans[out_dtype] = ovw.plan
            assert plans[torch.bfloat16] == plans[torch.float32], (label, proj)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_k256_tile_and_forced_forms_compose(arch, monkeypatch):
    """The 256-wide k tile composes (``"tile_k"``), and so do the 512-wide k tile's forms
    without an accumulate build (``main``, ``main_wave``) when the constants force them
    (``"stage"``); both keep the overwrite plan."""
    _fake_device(monkeypatch, arch, 148 if arch != "sm_107a" else 212, checks=False)
    for out_dtype in (torch.bfloat16, torch.float32):
        g, x, offs, out = _meta_operands(
            base.K256_WAVE_SIZES, *base.K256_WAVE_SHAPE, out_dtype
        )
        ovw, acc = _prepare_pair(g, x, offs, out)
        assert ovw.plan["tile_k"] == 256
        _assert_twin_launch(ovw, acc, arch)
        assert (acc.plan["accumulate"], acc.plan["accumulate_reason"]) == (
            "composed",
            "tile_k",
        )
    for form in ("main", "main_wave"):
        with pytest.MonkeyPatch.context() as mp:
            if not base._force_form(mp, form, arch):
                continue
            for out_dtype in (torch.bfloat16, torch.float32):
                k = base._form_ks(form)[0]
                g, x, offs, out = _meta_operands(base.K512_E3, 512, k, out_dtype)
                ovw, acc = _prepare_pair(g, x, offs, out)
                assert ovw.stages == (form,), (form, ovw.stages)
                _assert_twin_launch(ovw, acc, arch)
                assert (acc.plan["accumulate"], acc.plan["accumulate_reason"]) == (
                    "composed",
                    "stage",
                )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_two_launch_tail_never_fuses(arch):
    """An architecture whose 512-wide k tile takes neither one-launch tail (dropped from
    both ``ordered_tail_archs`` and ``unordered_tail_archs``) would sum its split tail
    with a second, overwriting tail-reduce launch, so its accumulate call composes
    (``"tail"``) although the stage's accumulate build is served and loadable; with the
    ordered one-launch tail (the shipped constants) the same stage is fused, and so it is
    with the unordered one (the architecture listed in ``unordered_tail_archs`` only)."""
    (route,) = SHIPPED_ACC_STAGES
    assert arch in HOST_PLAN_CONSTANTS["ordered_tail_archs"]
    assert arch not in HOST_PLAN_CONSTANTS["unordered_tail_archs"]
    two_launch = dict(
        HOST_PLAN_CONSTANTS, **_tail_lists(arch, ordered=False, unordered=False)
    )
    unordered = dict(
        HOST_PLAN_CONSTANTS, **_tail_lists(arch, ordered=False, unordered=True)
    )
    for stage in SHIPPED_ACC_STAGES[route][arch]:
        twin = stage[: -len(ACC)]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cb, "acc_library_loadable", lambda *args: True)
            assert cb.wgrad_accumulate_mode(
                route, twin, 512, arch, constants=two_launch
            ) == ("composed", "tail")
            assert cb.wgrad_accumulate_mode(route, twin, 512, arch) == ("fused", None)
            assert cb.wgrad_accumulate_mode(
                route, twin, 512, arch, constants=unordered
            ) == ("fused", None)


def test_plan_wgrad_takes_one_tail_form():
    """``plan_wgrad`` refuses both one-launch tails at once (``ValueError``) and records
    the form it plans: the ordered and the unordered one-launch tail plan the same split
    (one launch, no reduce grid, ``tail_splits`` slots per tail tile) and differ only in
    their form keys; neither form plans the two-launch tail (no slots)."""
    sizes = base.K512_E3
    n, k = base.CHANGED_SHAPES[0]
    common = dict(
        sum_m=sum(sizes),
        num_groups=len(sizes),
        n=n,
        k=k,
        tile_k=512,
        sm_count=148,
        reduce_threads=256,
        k256_tail=False,
        constants=HOST_PLAN_CONSTANTS,
    )
    with pytest.raises(ValueError, match="not both"):
        cb.plan_wgrad(**common, unordered_tail=True, ordered_tail=True)
    ordered = cb.plan_wgrad(**common, ordered_tail=True)
    unordered = cb.plan_wgrad(**common, unordered_tail=True)
    two_launch = cb.plan_wgrad(**common)
    assert ordered["ordered_tail"] and not ordered["unordered_tail"], ordered
    assert unordered["unordered_tail"] and not unordered["ordered_tail"], unordered
    forms = ("ordered_tail", "unordered_tail")
    assert _plan_without(ordered, *forms) == _plan_without(unordered, *forms)
    splits = ordered["tail_splits"]
    assert splits > 1 and ordered["launches"] == 1, ordered
    assert ordered["tail_reduce_grid"] is None, ordered
    slot = int(HOST_PLAN_CONSTANTS["cluster_m"]) * ordered["tail_tile_k"]
    assert ordered["tail_sum_elems"] == ordered["num_tail"] * splits * slot, ordered
    assert not two_launch["ordered_tail"] and not two_launch["unordered_tail"]
    assert two_launch["tail_sum_elems"] == 0 and two_launch["tail_ticket_words"] == 0


def _prepare_calls(g, x, offs, out):
    """The weight-gradient preparations of one call, overwrite and accumulate, through
    ``_prepare_wgrad`` and the public preparation."""
    return [
        lambda: cb._prepare_wgrad(g, x, offs, out, None, None),
        lambda: cb._prepare_wgrad(g, x, offs, out, None, None, accumulate=True),
        lambda: prepare_wgrad(g, x, offs, out=out),
        lambda: prepare_wgrad(g, x, offs, out=out, accumulate=True),
    ]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_preparation_refuses_an_architecture_in_both_tail_lists(arch, monkeypatch):
    """An architecture listed in both ``ordered_tail_archs`` and ``unordered_tail_archs``
    is refused by the weight-gradient preparation of a 512-wide k tile shape
    (``ValueError`` naming both lists, before anything is bound), overwrite and
    accumulate, into a bf16 and an fp32 ``out``."""
    _fake_device(monkeypatch, arch, 148 if arch != "sm_107a" else 212, checks=False)
    bound = []
    monkeypatch.setattr(cb, "_bind", lambda *args, **kwargs: bound.append(args))
    _set_tail_lists(monkeypatch, arch, ordered=True, unordered=True)
    sizes = base.K512_E3
    for n, k in base.CHANGED_SHAPES:
        tile_k = cb.select_wgrad_tile(
            k, arch, sum(sizes), len(sizes), HOST_PLAN_CONSTANTS
        )
        assert tile_k == 512, (n, k, tile_k)
        for out_dtype in (torch.bfloat16, torch.float32):
            g, x, offs, out = _meta_operands(sizes, n, k, out_dtype)
            for call in _prepare_calls(g, x, offs, out):
                with pytest.raises(ValueError, match="listed in both"):
                    call()
    assert bound == []


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_preparation_refuses_a_two_launch_tail_without_a_tail_reduce_stage(
    arch, monkeypatch
):
    """An architecture listed in neither ``ordered_tail_archs`` nor
    ``unordered_tail_archs`` takes the two-launch tail on the 512-wide k tile, whose
    routes serve no ``tail_reduce`` stage: the preparation of a ``main_halves`` weight
    gradient refuses it (``ValueError`` naming the route, before anything is bound),
    overwrite and accumulate, into a bf16 and an fp32 ``out``."""
    _fake_device(monkeypatch, arch, 148 if arch != "sm_107a" else 212, checks=False)
    bound = []
    monkeypatch.setattr(cb, "_bind", lambda *args, **kwargs: bound.append(args))
    assert base._force_form(monkeypatch, "main_halves", arch)
    _set_tail_lists(monkeypatch, arch, ordered=False, unordered=False)
    sizes = base.K512_E3
    n, k = 512, base._form_ks("main_halves")[0]
    tile_k = cb.select_wgrad_tile(k, arch, sum(sizes), len(sizes), HOST_PLAN_CONSTANTS)
    assert tile_k == 512, (k, tile_k)
    for out_dtype in (torch.bfloat16, torch.float32):
        route = cake_jit.record_name(
            "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=512
        )
        assert not cake_jit.stage_served(route, "tail_reduce", arch), route
        stage, _wave = cb.select_wgrad_stage(
            route,
            k,
            512,
            arch,
            HOST_PLAN_CONSTANTS,
            sum_m=sum(sizes),
            num_groups=len(sizes),
        )
        assert stage == "main_halves", stage
        g, x, offs, out = _meta_operands(sizes, n, k, out_dtype)
        for call in _prepare_calls(g, x, offs, out):
            with pytest.raises(ValueError, match="serves no tail_reduce stage"):
                call()
    assert bound == []


def test_preparation_refuses_a_one_launch_build_planned_as_two_launches(monkeypatch):
    """sm_107a's ``main_halves_k256t`` binds a one-launch split-tail build, while its
    two-launch tail would be summed by the 256-wide k tile's route, which does serve a
    ``tail_reduce`` stage: with sm_107a in neither tail list the preparation still
    refuses it (``ValueError`` naming the one-launch build, before anything is bound),
    overwrite and accumulate, into a bf16 and an fp32 ``out``."""
    arch = "sm_107a"
    _fake_device(monkeypatch, arch, 212, checks=False)
    bound = []
    monkeypatch.setattr(cb, "_bind", lambda *args, **kwargs: bound.append(args))
    assert base._force_form(monkeypatch, "main_halves_k256t", arch)
    _set_tail_lists(monkeypatch, arch, ordered=False, unordered=False)
    sizes = base.K512_E3
    n, k = 512, base._form_ks("main_halves_k256t")[0]
    for out_dtype in (torch.bfloat16, torch.float32):
        dtype = DTYPE_NAMES[out_dtype]
        route = cake_jit.record_name("wgrad", out_dtype=dtype, tile_k=512)
        reduce_route = cake_jit.record_name(
            "wgrad", out_dtype=dtype, tile_k=cb.K256_TAIL_TILE_K
        )
        assert cake_jit.stage_served(reduce_route, "tail_reduce", arch), reduce_route
        stage, _wave = cb.select_wgrad_stage(
            route,
            k,
            512,
            arch,
            HOST_PLAN_CONSTANTS,
            sum_m=sum(sizes),
            num_groups=len(sizes),
        )
        assert stage == "main_halves_k256t", stage
        g, x, offs, out = _meta_operands(sizes, n, k, out_dtype)
        for call in _prepare_calls(g, x, offs, out):
            with pytest.raises(ValueError, match="binds a one-launch split-tail build"):
                call()
    assert bound == []


def test_misaligned_out_composes_with_one_warning(monkeypatch):
    """An ``out`` whose base, row pitch or expert pitch is not 32-byte aligned takes the
    composed route (``"alignment"``) with the overwrite plan, and the first such
    preparation in the process warns once (``UserWarning`` naming the caller); an aligned
    ``out`` of the same call is fused."""
    arch = "sm_100a"
    _fake_device(monkeypatch, arch, 148)
    _force_k512(monkeypatch, arch)
    sizes, n, k = [64, 0, 64], 256, 512
    e = len(sizes)
    g, x, offs, aligned = _cpu_operands(sizes, n, k, torch.float32)
    ovw, acc = _prepare_pair(g, x, offs, aligned)
    _assert_twin_launch(ovw, acc, arch)
    assert _promised(arch, ovw.stages[0], torch.float32), ovw.stages
    assert acc.plan["accumulate"] == "fused" and "accumulate_reason" not in acc.plan
    flat = torch.zeros(e * (n + 1) * (k + 8) + 8, dtype=torch.float32)
    misaligned = {
        "base": flat[1 : 1 + e * n * k].view(e, n, k),
        "row pitch": flat.as_strided((e, n, k), (n * (k + 4), k + 4, 1)),
        "expert pitch": flat.as_strided((e, n, k), (n * k + 4, k, 1)),
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
    (``"min_cuda"``) with the overwrite plan and binds only the twin, whose program loads,
    while loading the accumulate build would raise the loader's ``min_cuda`` error; the
    first such preparation warns once.  ``generated_program_available(accumulate=True)``
    follows the same rule."""
    arch = "sm_103a"
    _fake_device(monkeypatch, arch, 152, loadable=False)
    _force_k512(monkeypatch, arch)
    (route,) = SHIPPED_ACC_STAGES
    acc_programs = {
        cake_jit.stage_binding(route, stage, arch)["program"]
        for stage in SHIPPED_ACC_STAGES[route][arch]
    }
    loaded = []

    def load(program, arch, specializations=None):
        if program in acc_programs:
            raise RuntimeError(
                f"{program} needs CUDA 13.4 or newer; the local toolkit is 13.0"
            )
        loaded.append(program)
        return base._NoModule()

    monkeypatch.setattr(cb, "load_module", load)
    sizes, n, k = [64, 0, 64], 256, 512
    g, x, offs, out = _cpu_operands(sizes, n, k, torch.float32)
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
    device = torch.device("cpu")
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


@pytest.mark.parametrize("name,stage,arch", _shipped_bindings())
def test_acc_library_loadable_is_the_loader_rule(name, stage, arch, monkeypatch):
    """``acc_library_loadable`` is the loader's own rule, decided without building,
    loading or probing anything: a prebuilt library under ``aot/`` loads whatever the
    toolkit; otherwise the JIT toolkit must reach the ``build.json`` entry's ``min_cuda``
    (13.4 for the fp32 accumulate builds); without FlashInfer's JIT nothing builds."""
    binding = cake_jit.stage_binding(name, stage, arch)
    library = cake_jit.library_name(
        binding["program"], arch, binding["specializations"]
    )
    floor = _cuda_version(cake_jit.cake_build.libraries_by_name()[library]["min_cuda"])
    assert floor == (13, 4), (library, floor)
    toolkit_of_process = cb._jit_cuda_version
    for loader in ("load_library", "jit_spec"):
        monkeypatch.setattr(cake_jit.cake_build, loader, _never_loads)
    for prebuilt, toolkit, want in (
        (True, None, True),
        (True, (12, 9), True),
        (False, None, False),
        (False, (13, 3), False),
        (False, (13, 4), True),
        (False, (13, 5), True),
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


@pytest.mark.parametrize("route", ["fused", "composed"])
def test_accumulate_refuses_unsafe_outputs_on_both_routes(route, monkeypatch):
    """``accumulate=True`` refuses (``ValueError``, before anything is bound) a missing
    ``out``, an ``out_dtype`` other than ``out``'s, an ``out`` of another dtype, rows or
    experts that overlap (an expanded ``out`` included) and an ``out`` that shares memory
    with ``g``, ``x`` or ``offs``, on the fused route's shape and on the composed route's
    alike; and a non-bool ``accumulate``."""
    arch = "sm_100a"
    _fake_device(monkeypatch, arch, 148)
    bound = []
    monkeypatch.setattr(
        cb, "_bind", lambda *args, **kwargs: bound.append(args) or base._NoModule()
    )
    sizes, n = [64, 0, 64], 256
    if route == "fused":
        _force_k512(monkeypatch, arch)
        k = 512
    else:
        k = 256  # the 256-wide k tile: composed
    g0, x0, offs0, _ = _cpu_operands(sizes, n, k, torch.float32)
    for _label, out, out_dtype, override, pattern in _reject_cases(sizes, n, k):
        g, x, offs = (
            override.get("g", g0),
            override.get("x", x0),
            override.get("offs", offs0),
        )
        for call in (
            lambda: cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
                g,
                x,
                offs,
                out=out,
                out_dtype=out_dtype,
                accumulate=True,
            ),
            lambda: pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
                g,
                x,
                offs,
                out=out,
                out_dtype=out_dtype,
                accumulate=True,
            ),
        ):
            with pytest.raises(ValueError, match=pattern):
                call()
    assert bound == []
    _, _, _, out = _cpu_operands(sizes, n, k, torch.float32)
    for value in (1, 0, None, "True"):
        with pytest.raises(ValueError, match="accumulate must be a bool"):
            cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
                g0, x0, offs0, out=out, accumulate=value
            )


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
    _force_k512(monkeypatch, arch)
    bound = []
    real = cb._bind

    def record(*args, **kwargs):
        bound.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(cb, "_bind", record)
    n, k = 256, 512
    g, x, offs, _ = _cpu_operands([128], n, k, torch.float32)
    out = torch.zeros(n * 2 * k).as_strided((1, n, k), (0, 2 * k, 1))
    span = (n - 1) * 2 * k + k
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "int32_elems", span - 1)
    with pytest.raises(ValueError, match="exceeds the 32-bit element addressing"):
        cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
            g, x, offs, out=out, accumulate=accumulate
        )
    assert bound == []
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "int32_elems", span)
    wide = out.as_strided(out.shape, (span + 1, 2 * k, 1))
    with pytest.raises(ValueError, match="exceeds the 32-bit element addressing"):
        cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
            g, x, offs, out=wide, accumulate=accumulate
        )
    assert bound == []
    launch = cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
        g, x, offs, out=out, accumulate=accumulate
    )
    assert launch.out is out
    assert launch.plan["accumulate"] == ("fused" if accumulate else "off")


def test_sum_m_zero_accumulates_nothing(monkeypatch):
    """No row: the accumulate preparation binds nothing, keeps no scratch and does not
    zero ``out`` (the overwrite zero-fills)."""
    arch = "sm_103a"
    _fake_device(monkeypatch, arch, 160)
    g, x, offs, out = _cpu_operands([0, 0], 256, 512, torch.float32)
    ovw, acc = _prepare_pair(g, x, offs, out)
    assert ovw.zero_fill and ovw.launches == ()
    assert acc.launches == () and acc.stages == ()
    assert not acc.zero_fill and acc.add_from is None and not acc.bump_version


@pytest.mark.parametrize("arch,sm_count", base.ARCH_SM_COUNTS)
def test_order_varying_follows_the_plan(arch, sm_count, monkeypatch):
    """A prepared weight gradient is ``order_varying`` (it alerts under torch's
    deterministic mode) exactly when its plan adds an unordered split tail
    (``unordered_tail`` and ``tail_splits > 1``) and it has a program to launch, in
    overwrite and accumulate mode alike (fused or composed).  Under the shipped
    constants the ``CHANGED_SHAPES`` on ``K512_E3`` split their ordered one-launch tail
    at every SM count and none of them varies; with the architecture listed in
    ``unordered_tail_archs`` instead of ``ordered_tail_archs`` (an override: no such
    build ships) the same calls plan the same split, unordered, and every one varies.
    The 256-wide k tile (two-launch ordered tail reduce) and no rows never vary."""
    _fake_device(monkeypatch, arch, sm_count, checks=False)
    shipped = {}
    for n, k in base.CHANGED_SHAPES:
        for out_dtype in (torch.bfloat16, torch.float32):
            g, x, offs, out = _meta_operands(base.K512_E3, n, k, out_dtype)
            for mode, launch in zip(
                ("overwrite", "accumulate"), _prepare_pair(g, x, offs, out), strict=True
            ):
                plan = launch.plan
                assert plan["ordered_tail"] and not plan["unordered_tail"], plan
                assert plan["tail_splits"] > 1, (n, k, out_dtype, mode, plan)
                assert not launch.order_varying, (n, k, out_dtype, mode, plan)
                shipped[n, k, out_dtype, mode] = plan
    with pytest.MonkeyPatch.context() as mp:
        _set_tail_lists(mp, arch, ordered=False, unordered=True)
        forms = ("ordered_tail", "unordered_tail")
        for (n, k, out_dtype, mode), want in shipped.items():
            g, x, offs, out = _meta_operands(base.K512_E3, n, k, out_dtype)
            launch = cb._prepare_wgrad(
                g, x, offs, out, None, None, accumulate=mode == "accumulate"
            )
            plan = launch.plan
            assert plan["unordered_tail"] and not plan["ordered_tail"], plan
            assert _plan_without(plan, *forms) == _plan_without(want, *forms), mode
            assert launch.order_varying, (n, k, out_dtype, mode, plan)
    for sizes, (n, k) in (
        (base.K256_WAVE_SIZES, base.K256_WAVE_SHAPE),
        ([0, 0], base.CHANGED_SHAPES[0]),
    ):
        g, x, offs, out = _meta_operands(sizes, n, k, torch.float32)
        for launch in _prepare_pair(g, x, offs, out):
            assert not launch.order_varying, (sizes, launch.plan)


def _recorded_acc_launch(monkeypatch, plan):
    """A composed accumulate ``CakeMoEGroupedGemmNondetLaunch`` whose program is a recorder and whose
    scratch holds ones: each launch adds one into ``out``; returns ``(launch, log)``."""
    monkeypatch.setattr(
        cb, "tvm_ffi", types.SimpleNamespace(use_torch_stream=contextlib.nullcontext)
    )
    log = []
    launch = cb.CakeMoEGroupedGemmNondetLaunch(
        op="wgrad",
        record_name="cake_moe_grouped_gemm_wgrad_f32_k512",
        stages=("main_halves",),
        plan={"accumulate": "composed", **plan},
        out=torch.zeros(4),
        launches=(cb._Launch(entry=lambda *args: log.append("launch"), arguments=()),),
        add_from=torch.ones(4),
    )
    return launch, log


def test_accumulate_alerts_only_before_an_order_varying_launch(monkeypatch):
    """Under torch's deterministic mode the accumulate entry points check ``out`` first
    and never alert while preparing (CPU operands: the ``out`` and device checks); an
    accumulate launch with an unordered split tail (the dormant alert: only an
    architecture listed in ``unordered_tail_archs`` plans one) alerts at ``launch()``
    before it runs anything (strict: ``RuntimeError``, ``out`` untouched; ``warn_only``:
    a ``UserWarning``, then it adds); one with a fixed order (an unsplit tail, or a split
    tail the ordered one-launch tail finishes) adds without an alert."""
    g, x, offs, _ = _cpu_operands([2, 2], 256, 64, torch.float32)
    with base._torch_deterministic():
        with pytest.raises(ValueError, match="pass the"):
            cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
                g, x, offs, accumulate=True
            )
        with pytest.raises(ValueError, match="pass the"):
            pkg.cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, accumulate=True)
    split, log = _recorded_acc_launch(
        monkeypatch, {"unordered_tail": True, "tail_splits": 2}
    )
    assert split.order_varying
    strict = pytest.raises(RuntimeError, match="CakeMoEGroupedGemmNondetLaunch.launch")
    with base._torch_deterministic(), strict:
        split.launch()
    assert log == [] and torch.equal(split.out, torch.zeros(4))
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            split.launch()
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    alerts = [w for w in rec if w.category is UserWarning]
    assert len(alerts) == 1 and "CakeMoEGroupedGemmNondetLaunch.launch" in str(
        alerts[0].message
    )
    assert log == ["launch"] and torch.equal(split.out, torch.ones(4))
    for plan in (
        {"unordered_tail": True, "tail_splits": 1},
        {"unordered_tail": False, "ordered_tail": True, "tail_splits": 2},
    ):
        fixed, log = _recorded_acc_launch(monkeypatch, plan)
        assert not fixed.order_varying, plan
        with base._torch_deterministic(), warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fixed.launch()
        assert [w for w in rec if w.category is UserWarning] == [], plan
        assert log == ["launch"] and torch.equal(fixed.out, torch.ones(4)), plan


def test_accumulate_keywords_and_signatures():
    """The new switches are keyword-only and off by default, in the package and in the
    facade, whose signatures stay the package's."""
    for fn, keyword in (
        (cb.cake_moe_grouped_gemm_nondet_train_prepare_wgrad, "accumulate"),
        (pkg.cake_moe_grouped_gemm_nondet_train_wgrad, "accumulate"),
        (cb._prepare_wgrad, "accumulate"),
        (cb.generated_program_available, "accumulate"),
        (pkg.cake_moe_grouped_gemm_nondet, "fuse_wgrad_accumulation"),
    ):
        parameter = inspect.signature(fn).parameters[keyword]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, fn
        assert parameter.default is False, fn
    forward = inspect.signature(
        cake_autograd.CakeMoEGroupedGemmNondetFunction.forward
    ).parameters
    assert forward["fuse_wgrad_accumulation"].default is False
    assert (
        "deterministic"
        not in inspect.signature(pkg.cake_moe_grouped_gemm_nondet).parameters
    )
    facade = pytest.importorskip("flashinfer.cake_moe_grouped_gemm_nondet_train")
    for name in (
        "cake_moe_grouped_gemm_nondet_train_wgrad",
        "cake_moe_grouped_gemm_nondet",
    ):
        impl = (
            pkg.cake_moe_grouped_gemm_nondet_train_wgrad
            if name == "cake_moe_grouped_gemm_nondet_train_wgrad"
            else pkg.cake_moe_grouped_gemm_nondet
        )
        assert inspect.signature(getattr(facade, name)) == inspect.signature(impl), name


def test_generated_program_available_with_accumulate(monkeypatch):
    """``generated_program_available(..., accumulate=True)`` is True exactly where the
    512-wide k tile route of the dtype serves an accumulate stage on the architecture
    whose library can load (either dtype when ``out_dtype`` is None); False for a dtype
    the weight gradient does not write, and for every dtype where no accumulate build can
    load; it refuses a non-bool switch and another operation."""
    device = torch.device("cpu")
    for loadable in (True, False):
        monkeypatch.setattr(cb, "acc_library_loadable", lambda *args, ok=loadable: ok)
        for arch in cake_jit.ARCHES:
            monkeypatch.setattr(cb, "arch_for", lambda d, arch=arch: arch)
            served = {}
            for out_dtype, name in DTYPE_NAMES.items():
                route = cake_jit.record_name("wgrad", out_dtype=name, tile_k=512)
                served[out_dtype] = bool(SHIPPED_ACC_STAGES.get(route, {}).get(arch))
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
        "vendored_acc_build_loader", root / "cake_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuda_version(text):
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def test_build_manifest_lists_the_acc_libraries(tmp_path):
    """``build.json`` has an entry for every shipped accumulate build (program,
    architecture, specializations), whose ``min_cuda`` is at least its twin's; the fp32
    builds (``WGRAD_OUT`` 32: the PTX ISA 9.4 ``.noftz.f32`` reduction) need CUDA 13.4.
    A vendored copy of the loader and the manifest lists them too."""
    libraries = cake_jit.cake_build.libraries_by_name()
    vendored = _load_vendored_build(tmp_path).libraries_by_name()
    found = set()
    for name, stage, arch in _shipped_bindings():
        binding = cake_jit.stage_binding(name, stage, arch)
        twin = cake_jit.stage_binding(name, stage[: -len(ACC)], arch)
        library = cake_jit.library_name(
            binding["program"], arch, binding["specializations"]
        )
        twin_library = cake_jit.library_name(
            twin["program"], arch, twin["specializations"]
        )
        assert library in libraries and library in vendored, library
        entry = libraries[library]
        assert entry["arch"] == arch
        floor = _cuda_version(entry["min_cuda"])
        assert floor >= _cuda_version(libraries[twin_library]["min_cuda"]), library
        assert binding["specializations"].get("WGRAD_OUT") == 32, (name, stage)
        assert floor == (13, 4), (library, entry["min_cuda"])
        found.add(library)
    assert len(found) == len(_shipped_bindings()) == 8


def test_fuse_flag_refusals_on_the_host():
    """``fuse_wgrad_accumulation`` must be a bool; with it, a ``w`` that requires grad
    needs a ``main_grad`` tensor of its shape, fp32 or bf16, on its device, with unit
    stride along the last dimension (``ValueError`` at the forward call, before any
    launch); a ``w`` that does not require grad needs none, and neither does a call
    under ``torch.no_grad()`` or inference mode, which autograd does not record."""
    e, n, k = 2, 256, 64
    x = torch.zeros(4, k, dtype=torch.bfloat16)
    offs = torch.tensor([2, 4], dtype=torch.int32)

    def call(w, flag=True):
        return pkg.cake_moe_grouped_gemm_nondet(
            x, w, offs, fuse_wgrad_accumulation=flag
        )

    w = torch.zeros(e, n, k, dtype=torch.bfloat16, requires_grad=True)
    for value in (1, None, "True"):
        with pytest.raises(ValueError, match="fuse_wgrad_accumulation must be a bool"):
            call(w, value)
        with pytest.raises(ValueError, match="fuse_wgrad_accumulation must be a bool"):
            cake_autograd.CakeMoEGroupedGemmNondetFunction.apply(
                x, w, offs, None, value
            )
    with pytest.raises(ValueError, match="no main_grad"):
        call(w)
    for bad in (
        torch.zeros(e, n, k + 1),
        torch.zeros(e, n, k, dtype=torch.float16),
        torch.zeros(e, n, 2 * k)[:, :, ::2],
        torch.zeros(e, n, k, device="meta"),
    ):
        w.main_grad = bad
        with pytest.raises(ValueError, match="w.main_grad must be"):
            call(w)
    frozen = torch.zeros(e, n, k, dtype=torch.bfloat16)
    with pytest.raises(
        ValueError
    ) as info:  # CPU operands: the device check, not main_grad
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
    with pytest.raises(NotImplementedError, match="fuse_wgrad_accumulation"):
        pkg.cake_moe_grouped_gemm_nondet(
            torch.zeros(4, 64, dtype=torch.bfloat16),
            w,
            torch.tensor([2, 4], dtype=torch.int32),
            fuse_wgrad_accumulation=True,
        )


def _det_tripwires(monkeypatch):
    """Replace the deterministic package's weight gradient and autograd wrapper by
    functions that fail the test when called: nothing may route there."""
    import flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train as det_pkg
    from flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_autograd as det_autograd,
    )

    def tripwire(*args, **kwargs):
        raise AssertionError("the facade routed to the deterministic package")

    monkeypatch.setattr(det_pkg, "cake_moe_grouped_gemm_det_train_wgrad", tripwire)
    monkeypatch.setattr(det_autograd, "cake_moe_grouped_gemm_det", tripwire)


def _facade_recorders(monkeypatch):
    """Replace the package functions the facade hands the weight gradient and the autograd
    wrapper to by recorders (and the deterministic ones by tripwires); return the log."""
    _det_tripwires(monkeypatch)
    log = []

    def new_wgrad(*args, **kwargs):
        log.append(("new_wgrad", kwargs))
        return "new"

    def new_mm(*args, **kwargs):
        log.append(("new_mm", kwargs))
        return "new"

    monkeypatch.setattr(pkg, "cake_moe_grouped_gemm_nondet_train_wgrad", new_wgrad)
    monkeypatch.setattr(cake_autograd, "cake_moe_grouped_gemm_nondet", new_mm)
    return log


def test_facade_accumulate_rows(monkeypatch):
    """The facade hands ``accumulate`` and ``fuse_wgrad_accumulation`` to the package,
    also under torch's deterministic mode: no reroute to the deterministic package, no
    warning, no ``NotImplementedError``."""
    facade = pytest.importorskip("flashinfer.cake_moe_grouped_gemm_nondet_train")
    log = _facade_recorders(monkeypatch)
    sizes, n, k = [2, 2], 256, 64
    g, x, offs, out = _cpu_operands(sizes, n, k, torch.float32)
    weight = torch.zeros(2, n, k, dtype=torch.bfloat16)
    for mode in (contextlib.nullcontext(), base._torch_deterministic()):
        with mode, warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            assert (
                facade.cake_moe_grouped_gemm_nondet_train_wgrad(
                    g, x, offs, out=out, accumulate=True
                )
                == "new"
            )
            assert log.pop() == (
                "new_wgrad",
                dict(
                    out=out,
                    out_dtype=None,
                    num_groups=None,
                    accumulate=True,
                ),
            )
            assert (
                facade.cake_moe_grouped_gemm_nondet(
                    x, weight, offs, fuse_wgrad_accumulation=True
                )
                == "new"
            )
            assert log.pop() == (
                "new_mm",
                dict(fuse_wgrad_accumulation=True),
            )
            assert log == []
        assert [r for r in rec if r.category is UserWarning] == []


@pytest.mark.parametrize("value", [True, False])
def test_facade_accumulate_rows_take_no_mode_keyword(value, monkeypatch):
    """The facade's accumulate rows take no ``deterministic`` keyword: either value
    raises Python's ``TypeError`` before the checks of ``out`` and before anything
    runs, also under torch's deterministic mode; the deterministic package is never
    called and ``out`` is untouched."""
    facade = pytest.importorskip("flashinfer.cake_moe_grouped_gemm_nondet_train")
    _det_tripwires(monkeypatch)
    g, x, offs, out = _cpu_operands([2, 2], 256, 64, torch.float32)
    weight = torch.zeros(2, 256, 64, dtype=torch.bfloat16)
    calls = (
        lambda: facade.cake_moe_grouped_gemm_nondet_train_wgrad(
            g, x, offs, out=out, accumulate=True, deterministic=value
        ),
        lambda: facade.cake_moe_grouped_gemm_nondet(
            x, weight, offs, deterministic=value, fuse_wgrad_accumulation=True
        ),
    )
    for mode in (contextlib.nullcontext(), base._torch_deterministic()):
        with mode:
            for call in calls:
                with pytest.raises(TypeError, match="deterministic"):
                    call()
    assert torch.equal(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


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


@contextlib.contextmanager
def _deterministic_mode(warn_only):
    """``torch.use_deterministic_algorithms(True, warn_only=warn_only)`` inside the block,
    the previous setting restored after it."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


def _rms(t):
    return float(t.double().pow(2).mean().sqrt().clamp_min(1e-30))


def _acc_route(device, form, out_dtype, monkeypatch):
    """Force the 512-wide k tile's ``form`` on the device (skip where the architecture has
    no such form) and return ``(arch, route)``: ``"fused"`` where this release ships the
    form's accumulate build for ``out_dtype`` on the architecture, else ``"composed"``.  A
    shipped build whose library cannot load here (neither prebuilt nor a JIT toolkit at
    its ``min_cuda``) skips with "cannot load" in the reason: that call composes
    (``"min_cuda"``), which the CPU tests and the composed GPU cases cover."""
    arch = base._arch(device)
    if not base._force_form(monkeypatch, form, arch):
        pytest.skip(f"no {form} build on {arch}")
    if not _promised(arch, form, out_dtype):
        return arch, "composed"
    name = cake_jit.record_name("wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=512)
    if not cb.acc_library_loadable(name, form + ACC, arch):
        pytest.skip(
            f"the {form}{ACC} library cannot load here (not prebuilt, and FlashInfer's JIT "
            "cannot build it at its build.json min_cuda): the call composes"
        )
    return arch, "fused"


def _require_fused(device, form, out_dtype, monkeypatch):
    """``_acc_route`` for a case of the fused route: this release must ship the build."""
    arch, route = _acc_route(device, form, out_dtype, monkeypatch)
    assert route == "fused", (
        f"no {form}{ACC} build for {out_dtype} on {arch} is shipped"
    )
    return arch


def _shape_with_splits(device, sizes, ks, split):
    """``(N, K)`` over ``ks`` (N up to 64 column blocks) whose weight-gradient plan on
    ``device`` takes the 512-wide k tile with ``tail_splits > 1`` (``split``) or 1, within
    the 32-bit element addressing; None when no candidate does."""
    if split:
        return base._split_tail_shape(device, sizes, ks)
    arch, sms = base._arch(device), cb.sm_count(device)
    span = int(HOST_PLAN_CONSTANTS["int32_elems"])
    for k in ks:
        for blocks in range(1, 65):
            n = 256 * blocks
            if len(sizes) * (n + 2) * (k + 32) > span:
                break
            _stage, plan = base._wgrad_plan(sizes, n, k, arch, sms)
            if plan["tile_k"] == 512 and plan["tail_splits"] == 1:
                return n, k
    return None


# The group sizes a split-tail case tries, in order: which sizes split a tail depends on
# the form and the SM count (``ACC_SIZES`` splits no k256t tail at 160 or 208 SMs).
SPLIT_SIZES = (ACC_SIZES,) + tuple([300 * m, 0, 1, 700 * m] for m in (4, 8, 16))


def _split_case(device, form):
    """``(sizes, n, k)``: the first group sizes of ``SPLIT_SIZES`` with a shape whose
    weight-gradient plan on ``device`` splits the ``form`` tail; None when none does."""
    for sizes in SPLIT_SIZES:
        shape = _shape_with_splits(device, sizes, base._form_ks(form), split=True)
        if shape is not None:
            return (list(sizes), *shape)
    return None


def _padded_out(e, n, k, dtype, device, fill_bits):
    """An ``[e, n, k]`` view of a row- and expert-padded buffer whose every element (the
    padding included) holds the bit pattern ``fill_bits``; returns ``(out, buffer)``."""
    buffer = torch.empty(e, n + 2, k + 32, dtype=dtype, device=device)
    _bits(buffer).fill_(fill_bits)
    return buffer[:, :n, :k], buffer


def _out0(kind, dw, sizes, dtype, gen):
    """The starting ``out`` of a case: ``"zeros"``, ``"randn"`` (randn x rms(dW)),
    ``"absorb"`` (randn x 2^12 x rms(dW)) or ``"specials"`` (randn x rms(dW) with -0.0,
    NaN and +-max/2^8 at tile and group boundaries); empty groups hold random bits with
    -0.0, a NaN payload and a subnormal in every case."""
    e, n, k = dw.shape
    scale = _rms(dw)
    if kind == "zeros":
        out0 = torch.zeros(e, n, k, device=dw.device)
    else:
        out0 = torch.randn(e, n, k, generator=gen, device=dw.device) * scale
        if kind == "absorb":
            out0 *= 2.0**12
    out0 = out0.to(dtype)
    if kind == "specials":
        big = torch.finfo(dtype).max / 2.0**8
        for group, rows in enumerate(sizes):
            if not rows:
                continue
            for (r, c), value in (
                ((0, 0), -0.0),
                ((0, 2), float("nan")),
                ((127, 128), big),
                ((128, 127), -big),
                ((255, 256), float("nan")),
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


def _overwrites(g, x, offs, out_dtype):
    """The overwrite weight gradients of the call in fp32 and in ``out_dtype`` (same plan)."""
    dw32 = prepare_wgrad(g, x, offs, out_dtype=torch.float32).launch().clone()
    dw = (
        dw32
        if out_dtype == torch.float32
        else prepare_wgrad(g, x, offs, out_dtype=out_dtype).launch().clone()
    )
    return dw32, dw


def _fused_oracle(out0, dw32):
    """The bits of the fused (fp32) accumulate on the groups with rows, every tile (whole,
    or split and finished by the ordered one-launch tail): one fp32 add of the overwrite
    of the same plan into ``out0`` (README numerical contract).  Compare it on the groups
    with rows only: on an empty group ``add_`` turns a -0.0 of ``out0`` into +0.0, while
    the fused build leaves ``out[e]`` untouched."""
    return out0.clone().add_(dw32)


def _expected(out0, g, x, sizes, times=1):
    """``(ref, scale)``: ``out0 + times * sum_e G[e].T X[e]`` in fp64 (as fp32), and the
    operands' magnitude ``|out0| + times * |dW|`` the bf16 budget scales with."""
    dw64 = base._reference("wgrad", g, x, sizes).double()
    start = out0.double()
    ref = (start + times * dw64).float()
    scale = (start.abs() + times * dw64.abs()).float()
    return ref, scale


def _assert_within_budget(out, expected, mask=None):
    """The package's budget of ``out``'s dtype against ``expected`` (:func:`_expected`)
    over ``mask`` (default all).  fp32: rel-L2 against the fp64 reference.  bf16: every
    element within ``1e-2 + 1e-2 * (|out0| + n |dW|)`` of it, the overwrite budget applied
    per term of the sum (an accumulate's rounding scales with its operands, not with a
    possibly cancelled result: AccumulateGrad itself misses a result-scaled budget)."""
    ref, scale = expected
    if mask is not None:
        out, ref, scale = out[mask], ref[mask], scale[mask]
    if out.dtype == torch.bfloat16:
        assert bool(torch.isfinite(out).all()), "a non-finite bf16 result"
        err = (out.double() - ref.double()).abs()
        bound = base.ATOL + base.RTOL * scale.double()
        bad = int((err > bound).sum())
        assert bad == 0, (
            f"bf16: {bad} elements beyond 1e-2 + 1e-2 (|out0| + n |dW|), worst "
            f"{float((err - bound).max()):.3e} over"
        )
        return
    rel_l2 = float(
        (out.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)
    )
    assert rel_l2 <= base.FP32_REL_L2, f"fp32 rel-L2 {rel_l2:.3e}"


def _assert_no_lost_update(out, out0, dw32, sizes):
    """fp32: every 128 x 128 block of a group with rows added its tile (``out - out0`` is
    ``dW`` within 2^-6 of the block's largest ``|dW|``), so no tile update was lost or
    applied twice."""
    e, n, k = out.shape
    delta = (out.double() - out0.double()).reshape(e, n // 128, 128, k // 128, 128)
    want = dw32.double().reshape(delta.shape)
    scale = want.abs().amax(dim=(2, 4), keepdim=True)
    finite = torch.isfinite(delta) & (out0.double().reshape(delta.shape).abs() < 1e30)
    bad = (
        (((delta - want).abs() > scale * 2.0**-6 + 1e-6) & finite).any(dim=4).any(dim=2)
    )
    for group, rows in enumerate(sizes):
        if not rows:
            bad[group] = False
    assert not bool(bad.any()), (
        f"lost or doubled tile updates at {bad.nonzero()[:4].tolist()}"
    )


def _special_mask(out0, sizes):
    """Elements of groups with rows whose ``out0`` is ordinary (no NaN, no +-max/2^8)."""
    mask = torch.isfinite(out0) & (out0.abs() < torch.finfo(out0.dtype).max / 2.0**9)
    for group, rows in enumerate(sizes):
        if not rows:
            mask[group] = False
    return mask


def _inputs_with_zero_columns(sizes, n, k, device, seed):
    """``base._inputs`` with x's columns 1 and k - 2 zero, so ``dW[:, :, 1]`` and
    ``dW[:, :, k - 2]`` are exact zeros (a subnormal ``out0`` there must be kept)."""
    x, w, g, offs = base._inputs(sizes, n, k, device, seed=seed)
    x[:, 1] = 0
    x[:, k - 2] = 0
    return x, w, g, offs


# ---------------------------------------------------------------------------
# C. Precision (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["zeros", "randn", "absorb", "specials"])
@pytest.mark.parametrize("out_dtype", FUSED_DTYPES)
@pytest.mark.parametrize("form", TWIN_FORMS)
def test_fused_accumulate_matches_its_oracles(form, out_dtype, kind, monkeypatch):
    """The fused route of every accumulate build the device's architecture ships, on an
    unsplit tail into a row- and expert-padded ``out``: bitwise ``out0.add_(dW32)`` with
    the overwrite of the same plan (README numerical contract; NaN payloads aside), within
    the budget of the fp64 ``out0 + dW`` on the ordinary elements, no lost tile update,
    subnormal ``out0`` kept where ``dW`` is zero, empty groups and the padding bitwise
    untouched, the same bits on a second run, and four accumulates within the budget of
    ``out0 + 4 dW``."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    _require_fused(device, form, out_dtype, monkeypatch)
    shape = _shape_with_splits(device, ACC_SIZES, base._form_ks(form), split=False)
    assert shape is not None, f"no candidate shape leaves the {form} tail unsplit"
    n, k = shape
    x, _w, g, offs = _inputs_with_zero_columns(ACC_SIZES, n, k, device, seed=31)
    dw32 = _overwrites(g, x, offs, out_dtype)[0]
    gen = torch.Generator(device=device).manual_seed(32)
    out0 = _out0(kind, dw32, ACC_SIZES, out_dtype, gen)
    if kind == "specials":
        for group, rows in enumerate(ACC_SIZES):
            if rows:
                out0[group, :, 1] = FP32_SUBNORMAL
    out, buffer = _padded_out(len(ACC_SIZES), n, k, out_dtype, device, 0x7FA5A5A5)
    out.copy_(out0)
    padding = _bits(buffer).clone()
    padding[:, :n, :k] = 0
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.plan["accumulate"] == "fused" and launch.stages == (form + ACC,)
    assert launch.plan["tail_splits"] == 1
    launch.launch()
    torch.cuda.synchronize()
    oracle = _fused_oracle(out0, dw32)
    rows = torch.tensor([r > 0 for r in ACC_SIZES], device=device)
    _assert_bitwise(out[rows], oracle[rows], f"{form} {out_dtype} {kind}")
    for group, count in enumerate(ACC_SIZES):
        if not count:
            assert torch.equal(_bits(out[group]), _bits(out0[group])), group
    rest = _bits(buffer).clone()
    rest[:, :n, :k] = 0
    assert torch.equal(rest, padding), "the padding was written"
    mask = _special_mask(out0, ACC_SIZES)
    _assert_within_budget(out, _expected(out0, g, x, ACC_SIZES), mask)
    if kind != "specials":
        _assert_no_lost_update(out, out0, dw32, ACC_SIZES)
    if kind == "specials":
        kept = out[:, :, 1][rows]
        assert bool((kept != 0).all()), "a subnormal out0 was flushed"
    first = out.clone()
    out.copy_(out0)
    launch.launch()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(first)), (
        "an unsplit fused accumulate changed run to run"
    )
    if kind == "randn":
        out.copy_(out0)
        for _ in range(4):
            launch.launch()
        torch.cuda.synchronize()
        _assert_within_budget(out, _expected(out0, g, x, ACC_SIZES, times=4), mask)
    print(f"FUSED form={form} dtype={out_dtype} kind={kind} n={n} k={k}")


@pytest.mark.parametrize("out_dtype", FUSED_DTYPES)
@pytest.mark.parametrize("form", TWIN_FORMS)
def test_fused_split_tail_matches_its_oracles(form, out_dtype, monkeypatch):
    """The fused route with a split tail (the ordered one-launch tail: not
    ``order_varying``): every repeat (reset by ``copy_``, the stream's tail slots
    garbage-filled first) is bitwise ``out0.add_(dW32)`` with the overwrite of the same
    plan on the groups with rows, split tail tiles included, and leaves the empty groups
    bitwise untouched, so every repeat has the same bits; the result is within the
    budget of the fp64 ``out0 + dW`` with no tile update lost, and the stream's tail
    tickets rest at zero."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    _require_fused(device, form, out_dtype, monkeypatch)
    case = _split_case(device, form)
    assert case is not None, f"no candidate shape splits the {form} tail"
    sizes, n, k = case
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=33)
    dw32 = _overwrites(g, x, offs, out_dtype)[0]
    gen = torch.Generator(device=device).manual_seed(34)
    out0 = _out0("randn", dw32, sizes, out_dtype, gen)
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    plan = launch.plan
    assert plan["accumulate"] == "fused" and launch.stages == (form + ACC,), plan
    assert plan["ordered_tail"] and plan["tail_splits"] > 1, plan
    assert not launch.order_varying, plan
    oracle = _fused_oracle(out0, dw32)
    rows = torch.tensor([r > 0 for r in sizes], device=device)
    for repeat in range(base.REPEATS):
        out.copy_(out0)
        assert base._poison_tail_sums(device), "no kept tail slots on this stream"
        launch.launch()
        torch.cuda.synchronize()
        base._assert_same_bits(out[rows], oracle[rows], f"{form} repeat {repeat}")
        for group, count in enumerate(sizes):
            if not count:
                assert torch.equal(_bits(out[group]), _bits(out0[group])), group
        base._assert_tail_at_rest(device)
    _assert_within_budget(out, _expected(out0, g, x, sizes), _special_mask(out0, sizes))
    _assert_no_lost_update(out, out0, dw32, sizes)


# (case, out dtype) of the composed route: the 256-wide k tile, the forced ``main`` /
# ``main_wave`` forms and a misaligned ``out`` for both dtypes; a half-release form for a
# dtype without accumulate builds; a shipped build whose library cannot load.
COMPOSED_CASES = (
    [
        (case, dtype)
        for case in ("k256", "main", "main_wave", "misaligned")
        for dtype in (torch.float32, torch.bfloat16)
    ]
    + [("halves", dtype) for dtype in DTYPE_NAMES if dtype not in FUSED_DTYPES]
    + [("unloadable", dtype) for dtype in FUSED_DTYPES]
)


@pytest.mark.parametrize(
    "case,out_dtype",
    COMPOSED_CASES,
    ids=[f"{c}-{DTYPE_NAMES[d]}" for c, d in COMPOSED_CASES],
)
def test_composed_accumulate_is_the_overwrite_plus_add(case, out_dtype, monkeypatch):
    """The composed route: the overwrite of the same plan into the launch's scratch, then
    ``out.add_``: bitwise ``out0.add_(dW)`` with the overwrite in ``out``'s dtype on an
    unsplit plan (for a bf16 ``out``, AccumulateGrad's bits), within the budget of the
    fp64 ``out0 + dW``; empty groups keep their values (``out.add_`` semantics).  The
    reason is the first condition of the fused route that fails: ``"tile_k"``,
    ``"stage"`` (the forced forms; every half-release form of a dtype without accumulate
    builds, misaligned or not), ``"min_cuda"`` or ``"alignment"``."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    monkeypatch.setattr(cb, "_WARNED", set())
    arch = base._arch(device)
    sizes = ACC_SIZES
    form = {"main": "main", "main_wave": "main_wave"}.get(case, "main_halves")
    if case == "k256":
        sizes, (n, k) = base.K256_WAVE_SIZES + [0], base.K256_WAVE_SHAPE
        reason = "tile_k"
    else:
        if not base._force_form(monkeypatch, form, arch):
            pytest.skip(f"no {form} build on {arch}")
        shape = _shape_with_splits(device, sizes, base._form_ks(form), split=False)
        assert shape is not None
        n, k = shape
        promised = _promised(arch, form, out_dtype)
        reason = "stage"
        if case == "unloadable":
            assert promised, (arch, out_dtype)
            monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: False)
            reason = "min_cuda"
        elif case == "misaligned" and promised:
            reason = "alignment"
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=35)
    dw = _overwrites(g, x, offs, out_dtype)[1]
    gen = torch.Generator(device=device).manual_seed(36)
    out0 = _out0("randn", dw.float(), sizes, out_dtype, gen)
    if case == "misaligned":
        flat = torch.empty(out0.numel() + 64, dtype=out_dtype, device=device)
        out = flat[1 : 1 + out0.numel()].view_as(out0)
    else:
        out = torch.empty_like(out0)
    out.copy_(out0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    plan = launch.plan
    assert (plan["accumulate"], plan["accumulate_reason"]) == ("composed", reason), plan
    if case != "k256":
        assert plan["tail_splits"] == 1
    launch.launch()
    torch.cuda.synchronize()
    _assert_bitwise(out, out0.clone().add_(dw), f"composed {case}")
    mask = _special_mask(out0, sizes)
    _assert_within_budget(out, _expected(out0, g, x, sizes), mask)
    for group, rows in enumerate(sizes):
        if not rows:
            same = (out[group] == out0[group]) | (
                torch.isnan(out[group]) & torch.isnan(out0[group])
            )
            assert bool(same.all()), group


def _differing_tiles(out, oracle, tile_k, sizes):
    """The ``(group, n block, k block)`` tiles (256 x ``tile_k``) of the groups with rows
    in which ``out`` and ``oracle`` differ in some bit (NaN payloads aside)."""
    e, n, k = out.shape
    nan = torch.isnan(out) & torch.isnan(oracle)
    differ = (_bits(out) != _bits(oracle)) & ~nan
    for group, rows in enumerate(sizes):
        if not rows:
            differ[group] = False
    tiles = differ.view(e, n // 256, 256, k // tile_k, tile_k).any(dim=4).any(dim=2)
    return tiles.nonzero().tolist()


BENCH_ROWS = [
    (label, sizes, proj)
    for label, sizes in base.BENCH_WGRAD_ROWS
    for proj in base.BENCH_PROJ
]


@pytest.mark.parametrize(
    "label,sizes,proj", BENCH_ROWS, ids=[f"{lab}-{p}" for lab, _s, p in BENCH_ROWS]
)
def test_fused_accumulate_on_the_benchmark_rows(label, sizes, proj, monkeypatch):
    """Every benchmark weight-gradient row (E4 balanced / skewed / one-empty at two token
    counts, E8 / E16 / E32 at 131072 rows) of both projections into an fp32 ``out``, as the
    planner plans it (no forced form): the call is fused (``plan["accumulate"] ==
    "fused"``, a shipped half-release accumulate build, with the ordered one-launch
    tail), within the fp64 budget of ``out0 + dW``, no tile update lost, an empty group
    bitwise untouched, and bitwise ``out0.add_(dW32)`` (the overwrite of the same plan)
    on every tile of the groups with rows, split tail tiles included (the stream's tail
    slots garbage-filled before the launch, its tickets at rest after it)."""
    out_dtype = torch.float32
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    arch = base._arch(device)
    n, k = base.BENCH_PROJ[proj]
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=61)
    dw32 = _overwrites(g, x, offs, out_dtype)[0]
    gen = torch.Generator(device=device).manual_seed(62)
    out0 = _out0("randn", dw32, sizes, out_dtype, gen)
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    plan = launch.plan
    twin = launch.stages[0].removesuffix(ACC)
    assert twin in TWIN_FORMS and _promised(arch, twin, out_dtype), (
        label,
        launch.stages,
    )
    route = cake_jit.record_name("wgrad", out_dtype="float32", tile_k=512)
    if not cb.acc_library_loadable(route, twin + ACC, arch):
        assert plan["accumulate_reason"] == "min_cuda", plan
        pytest.skip(f"the {twin}{ACC} library cannot load here: the call composes")
    assert plan["accumulate"] == "fused" and launch.stages == (twin + ACC,), plan
    assert plan["ordered_tail"] and not launch.order_varying, plan
    split = plan["tail_splits"] > 1
    # The overwrite of the same plan launched on this stream: it kept slots iff split.
    assert base._poison_tail_sums(device) == split, plan
    launch.launch()
    torch.cuda.synchronize()
    for group, rows in enumerate(sizes):
        if not rows:
            assert torch.equal(_bits(out[group]), _bits(out0[group])), group
    tiles = _differing_tiles(out, _fused_oracle(out0, dw32), plan["tail_tile_k"], sizes)
    assert not tiles, (
        f"{len(tiles)} tiles differ from out0.add_(dW32) (split tail: {plan['num_tail']} "
        f"tiles, {plan['tail_splits']} splits), e.g. {tiles[:4]}"
    )
    if split:
        base._assert_tail_at_rest(device)
    _assert_no_lost_update(out, out0, dw32, sizes)
    del dw32
    mask = _special_mask(out0, sizes)
    _assert_within_budget(out, _expected(out0, g, x, sizes), mask)
    print(
        f"BENCHROW {label} {proj} {arch} stage={launch.stages[0]} "
        f"splits={plan['tail_splits']} num_tail={plan['num_tail']} differing_tiles={len(tiles)}"
    )


@pytest.mark.parametrize(
    "label,sizes,proj", BENCH_ROWS, ids=[f"{lab}-{p}" for lab, _s, p in BENCH_ROWS]
)
def test_composed_bf16_accumulate_on_the_benchmark_rows(
    label, sizes, proj, monkeypatch
):
    """The same rows into a bf16 ``out``: the call composes (``"stage"``: no bf16
    accumulate build ships) with the overwrite's plan, and the result is within the
    operand-scaled bf16 budget of the fp64 ``out0 + dW``; an empty group keeps its
    values."""
    out_dtype = torch.bfloat16
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    n, k = base.BENCH_PROJ[proj]
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=63)
    gen = torch.Generator(device=device).manual_seed(64)
    scale = _rms(base._reference("wgrad", g[:4096], x[:4096], [4096]))
    out0 = (torch.randn(len(sizes), n, k, generator=gen, device=device) * scale).to(
        out_dtype
    )
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert (launch.plan["accumulate"], launch.plan["accumulate_reason"]) == (
        "composed",
        "stage",
    ), launch.plan
    launch.launch()
    torch.cuda.synchronize()
    for group, rows in enumerate(sizes):
        if not rows:
            assert torch.equal(out[group], out0[group]), group
    _assert_within_budget(out, _expected(out0, g, x, sizes), _special_mask(out0, sizes))


# ---------------------------------------------------------------------------
# D. Contract (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sizes", EMPTY_TAIL_SIZES, ids=["planned", "split_tail"])
@pytest.mark.parametrize("out_dtype", FUSED_DTYPES)
@pytest.mark.parametrize("form", TWIN_FORMS)
def test_fused_leaves_empty_groups_untouched(form, out_dtype, sizes, monkeypatch):
    """Empty groups placed where the k512 order puts them (the tail: planned walk engaged,
    split tail where a candidate shape splits it, the k256-tail halves) keep their bits
    on the fused route: random bits, -0.0, a NaN payload and a subnormal, compared as
    integers; the groups with rows are bitwise ``out0.add_(dW32)`` at every repeat (the
    stream's tail slots garbage-filled before each; the tickets at rest after a split
    tail) and within the budget."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    _require_fused(device, form, out_dtype, monkeypatch)
    ks = base._form_ks(form)
    shape = _shape_with_splits(device, sizes, ks, split=True) or _shape_with_splits(
        device, sizes, ks, split=False
    )
    assert shape is not None
    n, k = shape
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=37)
    dw32 = prepare_wgrad(g, x, offs, out_dtype=torch.float32).launch().clone()
    gen = torch.Generator(device=device).manual_seed(38)
    out0 = _out0("randn", dw32, sizes, out_dtype, gen)
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    plan = launch.plan
    assert plan["accumulate"] == "fused" and plan["ordered_tail"], plan
    assert not launch.order_varying, plan
    split = plan["tail_splits"] > 1
    oracle = _fused_oracle(out0, dw32)
    rows = torch.tensor([r > 0 for r in sizes], device=device)
    for repeat in range(2):
        out.copy_(out0)
        # The overwrite of this plan launched on this stream: it kept slots iff split.
        assert base._poison_tail_sums(device) == split, plan
        launch.launch()
        torch.cuda.synchronize()
        for group, count in enumerate(sizes):
            if not count:
                assert torch.equal(_bits(out[group]), _bits(out0[group])), group
        base._assert_same_bits(
            out[rows], oracle[rows], f"{form} {sizes} repeat {repeat}"
        )
        if split:
            base._assert_tail_at_rest(device)
    _assert_within_budget(out, _expected(out0, g, x, sizes), _special_mask(out0, sizes))
    print(
        f"EMPTY form={form} sizes={sizes} n={n} k={k} splits={launch.plan['tail_splits']}"
    )


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_no_rows_launch_nothing_and_leave_out(out_dtype):
    """``sum_m == 0``: the accumulate launch has no kernel and no add, and ``out`` keeps
    its bits (NaN padding included)."""
    device = base._require_program("wgrad", out_dtype)
    sizes, n, k = [0, 0], 256, 512
    x, _w, g, offs = base._inputs(sizes, n, k, device)
    out, buffer = _padded_out(
        2,
        n,
        k,
        out_dtype,
        device,
        0x7FC0 if out_dtype == torch.bfloat16 else 0x7FC00000,
    )
    before = _bits(buffer).clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.launches == () and launch.add_from is None
    assert launch.launch() is out
    torch.cuda.synchronize()
    assert torch.equal(_bits(buffer), before)


def _short_accumulate(device, groups, form, out_dtype, monkeypatch, seed):
    """An accumulate launch of short groups (``SMALL_E512``) or short beside long groups
    (``MIXED_E128``, split tail), the 512-wide k tile forced into ``form`` (fused where
    the release ships its accumulate build for ``out_dtype``, else composed) or as planned
    (``form`` None: the 256-wide k tile, composed).  Returns ``(launch, out0, sizes, g, x,
    offs)``."""
    arch = base._arch(device)
    sizes, n, k = base.SMALL_E512, 512, 3072
    want = "composed"
    if form is not None:
        arch, want = _acc_route(device, form, out_dtype, monkeypatch)
        if groups == "short":
            k = base._form_ks(form)[0]
        else:
            sizes = base.MIXED_E128
            shape = base._split_tail_shape(device, sizes, base._form_ks(form))
            assert shape is not None
            n, k = shape
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    out0 = (torch.randn(len(sizes), n, k, generator=gen, device=device)).to(out_dtype)
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.plan["accumulate"] == want, (arch, launch.plan)
    return launch, out0, sizes, g, x, offs


SHORT_ACC_CASES = [("short", None)] + [
    (groups, form) for groups in ("short", "mixed") for form in TWIN_FORMS
]


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "groups,form", SHORT_ACC_CASES, ids=[f"{g}-{f}" for g, f in SHORT_ACC_CASES]
)
def test_short_groups_accumulate_their_own_rows(groups, form, out_dtype, monkeypatch):
    """#6570 in accumulate mode: groups shorter than one k-step, alone or beside long ones
    with a split ordered tail, over ``SMALL_GROUP_LAUNCHES`` launches of one prepared
    call (each reset by ``copy_``, the descriptor slots filled with garbage first and the
    stream's tail slots before every launch): every 128 x 128 block of ``out - out0`` is
    the group's own ``dW``, and every launch has the bits of the first."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    launch, out0, sizes, g, x, _ = _short_accumulate(
        device, groups, form, out_dtype, monkeypatch, seed=39
    )
    plan = launch.plan
    assert not launch.order_varying, plan
    split = plan["ordered_tail"] and plan["tail_splits"] > 1
    if groups == "mixed":
        assert split, plan
    launch.workspaces[0].fill_(0xFF)
    e, n, k = out0.shape
    ref = torch.stack(
        [g[s:t].float().T @ x[s:t].float() for s, t in base._bounds(sizes)]
    )
    blocks = ref.view(e, n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True)
    tol = 2.0**-6 if out_dtype == torch.float32 else 2.0**-4
    wrong, changed, poisoned = [], [], 0
    first = None
    for call in range(base.SMALL_GROUP_LAUNCHES):
        launch.out.copy_(out0)
        poisoned += base._poison_tail_sums(device)
        got = launch.launch()
        out = got.float() - out0.float()
        bad = (
            ((out.view_as(blocks) - blocks).abs() > scale * tol + 2.0**-6)
            .any(dim=4)
            .any(dim=2)
        )
        if bool(bad.any()):
            wrong.append((call, bad.nonzero()[:4].tolist()))
        if first is None:
            first = got.clone()
        elif not torch.equal(_bits(got), _bits(first)):
            changed.append(call)
    assert not wrong, f"{len(wrong)} launches have wrong blocks: {wrong[:3]}"
    assert not changed, (
        f"{len(changed)} launches differ from the first in some bit: {changed[:8]}"
    )
    if split:
        # Every launch after the first found the stream's kept slots to poison.
        assert poisoned == base.SMALL_GROUP_LAUNCHES - 1, poisoned
        torch.cuda.synchronize()
        base._assert_tail_at_rest(device)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("form", TWIN_FORMS)
def test_accumulate_on_two_concurrent_streams(form, out_dtype, monkeypatch):
    """Two prepared accumulates (fused where the release ships the form's accumulate build
    for the dtype, else composed) of the mixed short / long groups (split ordered tail)
    into separate outputs, launched interleaved on two streams (each stream's tail slots
    garbage-filled before every launch) while a kernel stays resident on a third and
    copies and GEMMs run on a fourth: every result has the bits of an isolated launch on
    the quiet current stream, which is within the budget of ``out0 + dW``; each stream
    keeps its own tail slots and tickets, and the tickets rest at zero."""
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    launch, out0, sizes, g, x, offs = _short_accumulate(
        device, "mixed", form, out_dtype, monkeypatch, seed=41
    )
    second = prepare_wgrad(g, x, offs, out=out0.clone(), accumulate=True)
    launches = [launch, second]
    assert second.stages == launch.stages
    for item in launches:
        plan = item.plan
        assert plan["ordered_tail"] and plan["tail_splits"] > 1, plan
        assert not item.order_varying, plan
    launch.out.copy_(out0)
    isolated = launch.launch().clone()
    torch.cuda.synchronize()
    _assert_within_budget(
        isolated, _expected(out0, g, x, sizes), _special_mask(out0, sizes)
    )
    streams = [torch.cuda.Stream(device) for _ in range(4)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    src = torch.randn(1 << 24, device=device)
    dst = torch.empty_like(src)
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    with torch.cuda.stream(streams[2]):
        torch.cuda._sleep(base.RESIDENT_SLEEP_CYCLES)
    results = [[], []]
    for _ in range(8):
        for index in range(2):
            base._poison_tail_sums(device, streams[index])
            with torch.cuda.stream(streams[index]):
                launches[index].out.copy_(out0)
                results[index].append(launches[index].launch().clone())
        with torch.cuda.stream(streams[3]):
            dst.copy_(src)
            torch.mm(a, a, out=c)
    for stream in streams:
        main.wait_stream(stream)
    torch.cuda.synchronize()
    for index in range(2):
        for call, out in enumerate(results[index]):
            base._assert_same_bits(out, isolated, f"stream {index} launch {call}")
    for stream in streams[:2]:
        base._assert_tail_at_rest(device, stream)
    kept = [cb.stream_workspace("tail_sums", device, stream) for stream in streams[:2]]
    assert kept[0].data_ptr() != kept[1].data_ptr(), "two streams share tail slots"


@pytest.mark.parametrize("route", ["fused", "composed"])
def test_graph_replay_accumulates_at_every_replay(route, monkeypatch):
    """A captured accumulate launch adds at every replay: two replays without a reset
    have the bits of two eager launches of the same prepared call from ``out0`` (the
    fused route on a split ordered tail, the eager stream's tail slots garbage-filled
    first) and are within the budget of ``out0 + 2 dW``, also after new group boundaries
    are written into ``offs`` (the same ``sum_m``) and after the first ones are written
    back; the composed route's add is captured with its kernel."""
    out_dtype = torch.float32
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    if route == "fused":
        _require_fused(device, "main_halves", out_dtype, monkeypatch)
        sizes = list(base.K512_E3)
        shape = base._split_tail_shape(device, sizes, base._form_ks("main_halves"))
        assert shape is not None
        n, k = shape
        second = [sizes[0] // 2, sizes[2], sizes[0] - sizes[0] // 2]
    else:
        sizes, (n, k) = list(base.K256_WAVE_SIZES), base.K256_WAVE_SHAPE
        second = [1024, 3072] + sizes[2:]
    assert sum(second) == sum(sizes) and len(second) == len(sizes)
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=43)
    gen = torch.Generator(device=device).manual_seed(44)
    out0 = torch.randn(len(sizes), n, k, generator=gen, device=device)
    out = out0.clone()
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    plan = launch.plan
    assert plan["accumulate"] == route and not launch.order_varying, plan
    if route == "fused":
        assert plan["ordered_tail"] and plan["tail_splits"] > 1, plan
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        launch.launch()  # the first launch on a stream outside the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    for index, layout in enumerate((sizes, second, sizes)):
        offs.copy_(base._offs(layout, device))
        out.copy_(out0)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        out.copy_(out0)
        for _ in range(2):
            base._poison_tail_sums(device)
            launch.launch()
        torch.cuda.synchronize()
        base._assert_same_bits(
            replayed, out, f"{route} layout {index}: replay vs eager"
        )
        _assert_within_budget(replayed, _expected(out0, g, x, layout, times=2))


@pytest.mark.parametrize("route", ["fused", "composed"])
def test_prepared_accumulate_allocates_nothing_after_the_first_launch(
    route, monkeypatch
):
    """The composed scratch is allocated by the preparation; after the first launch on a
    stream neither route allocates device memory."""
    out_dtype = torch.float32
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    if route == "fused":
        _require_fused(device, "main_halves", out_dtype, monkeypatch)
        shape = base._split_tail_shape(
            device, base.K512_E3, base._form_ks("main_halves")
        )
        assert shape is not None
        sizes, (n, k) = base.K512_E3, shape
    else:
        sizes, (n, k) = base.K256_WAVE_SIZES, base.K256_WAVE_SHAPE
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=45)
    out = torch.zeros(len(sizes), n, k, device=device)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
    assert launch.plan["accumulate"] == route
    torch.cuda.synchronize()
    if route == "composed":
        assert launch.add_from is not None
        assert (
            torch.cuda.memory_allocated(device) - before >= launch.add_from.numel() * 4
        )
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
    ``torch.autograd.graph.increment_version``, the composed one through ``out.add_``);
    an inference tensor is accumulated into without an error."""
    out_dtype = torch.float32
    device = base._require_program("wgrad", out_dtype)
    if route == "fused":
        _require_fused(device, "main_halves", out_dtype, monkeypatch)
        sizes, n, k = ACC_SIZES, 256, 1024
    else:
        sizes, (n, k) = base.K256_WAVE_SIZES, base.K256_WAVE_SHAPE
    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=46)
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
    inference = prepare_wgrad(g, x, offs, out=frozen, accumulate=True)
    inference.launch()
    torch.cuda.synchronize()
    _assert_within_budget(frozen, _expected(torch.zeros_like(frozen), g, x, sizes))


# ---------------------------------------------------------------------------
# E. Autograd (GPU)
# ---------------------------------------------------------------------------


def _autograd_case(out_dtype, monkeypatch, form="main_halves"):
    """``(device, sizes, x0, w0, g1, g2, offs)`` of an unsplit 512-wide k tile shape in
    ``form``: its backward adds into an fp32 ``main_grad`` with the shipped accumulate
    build (fused) and into a bf16 one by composing (no bf16 accumulate build ships)."""
    base._require_program("fwd")
    base._require_program("dgrad")
    device = base._require_program("wgrad", out_dtype)
    base._fresh_stream_workspaces(monkeypatch)
    _arch, route = _acc_route(device, form, out_dtype, monkeypatch)
    assert route == ("fused" if out_dtype in FUSED_DTYPES else "composed")
    shape = _shape_with_splits(device, ACC_SIZES, base._form_ks(form), split=False)
    assert shape is not None
    n, k = shape
    x0, w0, g1, offs = base._inputs(ACC_SIZES, n, k, device, seed=51)
    g2 = base._inputs(ACC_SIZES, n, k, device, seed=52)[2]
    return device, ACC_SIZES, x0, w0, g1, g2, offs


def _composed_main_grad(mg0, grads, x, offs):
    """``mg0`` plus each weight gradient of ``grads`` added in ``mg0``'s dtype (the composed
    contract: the overwrite in that dtype, then ``add_``)."""
    mg = mg0.clone()
    for g in grads:
        mg.add_(
            pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
                g, x, offs, out_dtype=mg0.dtype
            )
        )
    return mg


def _fused_gemm(offs):
    return lambda a, b: pkg.cake_moe_grouped_gemm_nondet(
        a, b, offs, fuse_wgrad_accumulation=True
    )


def _run_micro_steps(fn, x0, w0, mg0, grads, ckpt, early_stop, attribute=None):
    """Micro-steps of ``fn(x, w)`` with ``w.main_grad = mg0.clone()``; returns ``(w, x
    grads per step)``."""
    w = w0.clone().requires_grad_(True)
    w.main_grad = mg0.clone()
    if attribute is not None:
        for key, value in attribute.items():
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
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_autograd_accumulates_into_main_grad(out_dtype, ckpt, early_stop, monkeypatch):
    """Two micro-steps with ``fuse_wgrad_accumulation=True`` (plain, non-reentrant and
    nested checkpoints, early stop on and off) leave ``main_grad == mg0 + dW1 + dW2``:
    bitwise the composed adds on this unsplit shape (the fused fp32 route; a bf16
    ``main_grad`` composes, AccumulateGrad's bits), within the budget of the fp64 sum;
    ``x.grad`` is bitwise the default mode's and ``w.grad`` stays None (no
    ``grad_added_to_main_grad``)."""
    device, sizes, x0, w0, g1, g2, offs = _autograd_case(out_dtype, monkeypatch)
    gen = torch.Generator(device=device).manual_seed(53)
    mg0 = torch.randn(w0.shape, generator=gen, device=device).to(out_dtype)
    w, dxs = _run_micro_steps(
        _fused_gemm(offs), x0, w0, mg0, (g1, g2), ckpt, early_stop
    )
    assert w.grad is None
    want = _composed_main_grad(mg0, (g1, g2), x0, offs)
    _assert_bitwise(w.main_grad, want, "main_grad")
    dws = [base._reference("wgrad", g, x0, sizes).double() for g in (g1, g2)]
    start = mg0.double()
    expected = (
        (start + dws[0] + dws[1]).float(),
        (start.abs() + dws[0].abs() + dws[1].abs()).float(),
    )
    _assert_within_budget(w.main_grad, expected)
    _, default_dxs = _run_micro_steps(
        lambda a, b: pkg.cake_moe_grouped_gemm_nondet(a, b, offs),
        x0,
        w0,
        mg0,
        (g1, g2),
        ckpt,
        early_stop,
    )
    for got, expect in zip(dxs, default_dxs, strict=True):
        assert torch.equal(got, expect)


@pytest.mark.parametrize("zero_out", [False, True], ids=["empty", "zeros"])
def test_autograd_returns_megatron_placeholder(zero_out, monkeypatch):
    """With ``w.grad_added_to_main_grad``: the backward sets it, adds into ``main_grad``
    and returns a fresh gradient of ``w``'s shape and dtype (zeros under
    ``zero_out_wgrad``), which AccumulateGrad stores as ``w.grad``."""
    device, sizes, x0, w0, g1, g2, offs = _autograd_case(torch.float32, monkeypatch)
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


def test_graph_captures_the_fused_backward(monkeypatch):
    """The forward and the fused backward of ``cake_moe_grouped_gemm_nondet(...,
    fuse_wgrad_accumulation=True)`` capture into one CUDA graph (the backward binds the
    context without querying the captured stream): on an unsplit shape (one writer per
    element) ``r`` replays from ``mg0`` give bitwise ``mg0`` plus ``r`` composed adds of
    ``dW``, and ``x.grad`` holds the eager backward's bits."""
    device, _sizes, x0, w0, g1, _g2, offs = _autograd_case(torch.float32, monkeypatch)
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
    for replays in (1, 3):
        w.main_grad.copy_(mg0)
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        want = _composed_main_grad(mg0, (g1,) * replays, x0, offs)
        _assert_bitwise(w.main_grad, want, f"{replays} replays")
        assert torch.equal(x.grad, eager_dx)


def test_autograd_flag_errors_and_side_effects(monkeypatch):
    """Missing ``main_grad`` at the forward and a wrong one at the backward raise
    ``ValueError``; ``w`` as a reentrant checkpoint's input arrives without ``main_grad``
    (``ValueError``); ``retain_graph`` twice adds twice; a frozen ``w`` leaves
    ``main_grad``; ``create_graph`` raises ``NotImplementedError``; a
    ``torch.autograd.grad`` whose ``inputs`` exclude ``w`` still adds into ``main_grad``;
    under torch's deterministic mode this unsplit (fixed-order) shape runs without an
    alert and adds the same bits."""
    device, sizes, x0, w0, g1, _g2, offs = _autograd_case(torch.float32, monkeypatch)
    gemm = _fused_gemm(offs)
    mg0 = torch.zeros(w0.shape, device=device)
    dw1 = pkg.cake_moe_grouped_gemm_nondet_train_wgrad(
        g1, x0, offs, out_dtype=torch.float32
    )

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
    (dx,) = torch.autograd.grad(gemm(x, w), (x,), g1)
    torch.cuda.synchronize()
    _assert_bitwise(w.main_grad, mg0.clone().add_(dw1), "autograd.grad(inputs=(x,))")
    x, w = leaves()
    y = gemm(x, w)
    with base._torch_deterministic(), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        y.backward(g1)
    torch.cuda.synchronize()
    assert [w for w in rec if w.category is UserWarning] == []
    _assert_bitwise(w.main_grad, mg0.clone().add_(dw1), "torch deterministic mode")


@pytest.mark.parametrize("form", TWIN_FORMS)
def test_autograd_fuse_never_alerts_on_a_split_tail(form, monkeypatch):
    """With ``fuse_wgrad_accumulation=True`` on a shape whose ``form`` splits its tail
    (the ordered one-launch tail, fused: not ``order_varying``), the forward and the
    backward under torch's strict and ``warn_only`` deterministic modes raise no alert
    and warn nothing, set the placeholder flag and leave ``main_grad`` and ``x.grad``
    with the bits of the run with the mode off (the stream's tail slots garbage-filled
    before every backward); that ``main_grad`` is bitwise the composed add
    (``mg0.add_(dW32)``) on the groups with rows, keeps ``mg0``'s bits on the empty
    groups and is within the budget of the fp64 sum."""
    base._require_program("fwd")
    base._require_program("dgrad")
    device = base._require_program("wgrad", torch.float32)
    base._fresh_stream_workspaces(monkeypatch)
    _arch, route = _acc_route(device, form, torch.float32, monkeypatch)
    assert route == "fused", (form, route)
    case = _split_case(device, form)
    assert case is not None, f"no candidate shape splits the {form} tail"
    sizes, n, k = case
    x0, w0, g, offs = base._inputs(sizes, n, k, device, seed=58)
    gen = torch.Generator(device=device).manual_seed(59)
    mg0 = torch.randn(w0.shape, generator=gen, device=device)
    # The backward prepares this accumulate (same operands, same plan).
    planned = prepare_wgrad(g, x0, offs, out=mg0.clone(), accumulate=True)
    plan = planned.plan
    assert plan["accumulate"] == "fused" and planned.stages == (form + ACC,), plan
    assert plan["ordered_tail"] and plan["tail_splits"] > 1, plan
    assert not planned.order_varying, plan
    gemm = _fused_gemm(offs)

    def run(mode):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        w.main_grad = mg0.clone()
        w.grad_added_to_main_grad = False
        with mode, warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            y = gemm(x, w)
            base._poison_tail_sums(device)
            y.backward(g)
        torch.cuda.synchronize()
        alerts = [r for r in rec if r.category is UserWarning]
        assert alerts == [], [str(r.message) for r in alerts]
        assert w.grad_added_to_main_grad is True and w.grad is not None
        return w.main_grad, x.grad

    want = run(contextlib.nullcontext())
    for warn_only in (False, True):
        main_grad, dx = run(_deterministic_mode(warn_only))
        base._assert_same_bits(main_grad, want[0], f"main_grad warn_only={warn_only}")
        base._assert_same_bits(dx, want[1], f"x.grad warn_only={warn_only}")
    rows = torch.tensor([r > 0 for r in sizes], device=device)
    composed = _composed_main_grad(mg0, (g,), x0, offs)
    base._assert_same_bits(
        want[0][rows], composed[rows], "main_grad vs the composed add"
    )
    for group, count in enumerate(sizes):
        if not count:
            assert torch.equal(_bits(want[0][group]), _bits(mg0[group])), group
    dw = base._reference("wgrad", g, x0, sizes).double()
    expected = (
        (mg0.double() + dw).float(),
        (mg0.double().abs() + dw.abs()).float(),
    )
    _assert_within_budget(want[0], expected)


# ---------------------------------------------------------------------------
# Sanitizer runner (compute-sanitizer wraps ``python <this file> sanitize``)
# ---------------------------------------------------------------------------


def _sanitize():
    """One small accumulate launch of every accumulate build this release ships for the
    device's architecture, with an unsplit and a split tail, checked within the budget;
    fails when such a build is missing or cannot load."""
    device = torch.device("cuda", 0)
    arch = base._arch(device)
    ran, missing = 0, []
    for out_dtype in FUSED_DTYPES:
        for form in TWIN_FORMS:
            if not _promised(arch, form, out_dtype):
                continue
            name = cake_jit.record_name(
                "wgrad", out_dtype=DTYPE_NAMES[out_dtype], tile_k=512
            )
            assert cake_jit.stage_served(name, form + ACC, arch), (form, arch)
            assert cb.acc_library_loadable(name, form + ACC, arch), (form, arch)
            for split in (False, True):
                with pytest.MonkeyPatch.context() as mp:
                    if not base._force_form(mp, form, arch):
                        continue
                    # The small groups never split a tail; the split case takes
                    # the split-tail tests' candidates.
                    if split:
                        case = _split_case(device, form)
                    else:
                        sizes = [300, 0, 1, 700]
                        shape = _shape_with_splits(
                            device, sizes, base._form_ks(form), split
                        )
                        case = None if shape is None else (sizes, *shape)
                    if case is None:
                        print(
                            f"SANITIZE skip {form} {out_dtype} split={split}: no shape"
                        )
                        missing.append((form, out_dtype, split))
                        continue
                    sizes, n, k = case
                    x, _w, g, offs = base._inputs(sizes, n, k, device, seed=57)
                    out0 = torch.randn(len(sizes), n, k, device=device).to(out_dtype)
                    out = out0.clone()
                    launch = prepare_wgrad(g, x, offs, out=out, accumulate=True)
                    assert launch.plan["accumulate"] == "fused", launch.plan
                    assert (launch.plan["tail_splits"] > 1) == split, launch.plan
                    launch.launch()
                    torch.cuda.synchronize()
                    _assert_within_budget(
                        out, _expected(out0, g, x, sizes), _special_mask(out0, sizes)
                    )
                    ran += 1
                    print(
                        f"SANITIZE {arch} {form}{ACC} {out_dtype} n={n} k={k} "
                        f"splits={launch.plan['tail_splits']} ok"
                    )
    print(f"SANITIZE cases={ran} missing={len(missing)}")
    return 0 if ran and not missing else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["sanitize"]:
        sys.exit(_sanitize())
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
