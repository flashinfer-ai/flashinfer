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

Host-plan, packaging and mode-policy tests of the Cake four-stream mHC training
package with the non-deterministic backward
(``flashinfer.experimental.cake.cake_mhc_train_backend``; FlashInfer issue
#8068; no GPU needed, nothing is built or launched).

The package is a standalone copy of the deterministic package
(``flashinfer.cake_mhc_det_train_backend``) under the ``cake_mhc_train_``
prefix, except the ``bwd_finish`` route.  This module checks the registry
``cake_jit.py`` against the deterministic registry on its boards of record
(``RULES_BOARDS``) and token counts of record (``RULES_T_VALUES``):

* the program set is the deterministic set without its column-slab
  (``bwd_slab__*``) and final-reduction (``bwd_reduce__*``) records, renamed
  and otherwise equal field for field, plus the three programs of this package
  (``cake_mhc_train_fn_split_atomic__tv128_dep`` on sm_100a / sm_103a,
  ``cake_mhc_train_fn_split_atomic__tv128_t128_dep`` on sm_107a and
  ``cake_mhc_train_bwd_slab_atomic__dynr`` on all three), whose records keep
  the launch, grid and roles of the deterministic records they replace (the
  fn split with the static PDL attribute: it follows the token-owner pass);
* no record, route, record scalar or PDL edge names a deterministic library;
* every route, the selection on every route and the PDL bands equal the
  deterministic ones, except ``bwd_finish``: the deterministic token-owner
  records, the atomic fn split of the architecture, the atomic column slab, no
  reduction;
* the workspaces: ``fwd`` the deterministic one, ``bwd`` the deterministic
  regions at the same offsets without the trailing ``dfn_part``;
* the host rules are copies (equal outputs, not the deterministic objects);
* the package is standalone: no source file imports another flashinfer package
  (the shared ``cake_build.py`` JIT fallback aside, inside a function), and a
  copy under another name imports and plans with ``flashinfer`` blocked;
* the mode policy without a device: every compute and workspace entry point
  refuses any ``deterministic`` other than ``False`` first (``ValueError``),
  only ``backward_finish`` raises PyTorch's alert under
  ``torch.use_deterministic_algorithms(True)`` (before any validation or
  launch), and the alert's wording;
* the wiring: the test-helper key, the package data, the build-manifest
  checker's row and ``build.json``.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import PACKAGES, cake_package

KEY = "cake.cake_mhc_train"
PACKAGE_NAME = "flashinfer.experimental.cake.cake_mhc_train_backend"
pkg = cake_package(KEY)
nd = cake_package(KEY, "cake_jit")
cake_build = cake_package(KEY, "cake_build")
det = cake_package("cake_mhc_det_train", "cake_jit")
det_build = cake_package("cake_mhc_det_train", "cake_build")

PACKAGE_DIR = Path(nd.__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
HIDDEN = 4096
ND_PREFIX, DET_PREFIX = "cake_mhc_train_", "cake_mhc_det_train_"
BWD_FINISH = "bwd_finish_h4096"
# The programs of this package and their architectures.
ATOMIC = {
    "cake_mhc_train_fn_split_atomic__tv128_dep": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fn_split_atomic__tv128_t128_dep": ("sm_107a",),
    "cake_mhc_train_bwd_slab_atomic__dynr": ("sm_100a", "sm_103a", "sm_107a"),
}
SLAB = "cake_mhc_train_bwd_slab_atomic__dynr"
DET_DYNR = "cake_mhc_det_train_bwd_slab__dynr"
# The deterministic programs this package does not copy: the column slab and the final reduction (the backward fn
# splits, whose libraries the forward does not share, have no copy either).
DROPPED = (DET_PREFIX + "bwd_slab__", DET_PREFIX + "bwd_reduce__")
DROPPED_ROLES = ("bwd_slab", "bwd_reduce")
# Sinkhorn counts other than the count of record: the token-owner pass runs its run-time-count record.
RUN_TIME_ITERS = (1, 7, 64)
T_VALUES = tuple(T for T in det.RULES_T_VALUES if T > 0)
# The public entry points that take the mode keyword (every compute and workspace entry point).
MODE_ENTRIES = (
    "cake_mhc_train_forward_prepare",
    "cake_mhc_train_forward_stats",
    "cake_mhc_train_forward_finish",
    "cake_mhc_train_backward_prepare",
    "cake_mhc_train_backward_finish",
    "cake_mhc_train_prepare",
    "cake_mhc_train_finish",
    "cake_mhc_train_forward_workspace_bytes",
    "cake_mhc_train_backward_workspace_bytes",
    "cake_mhc_train_forward_workspace",
    "cake_mhc_train_backward_workspace",
)
# The public functions without it.
MODELESS = (
    "is_cake_mhc_train_supported",
    "cake_mhc_train_reset_workspace",
    "route_key",
    "route_modules",
)
_ALERT = "does not have a deterministic implementation"
_REFUSAL = "computes non-deterministically and takes deterministic=False"


def _renamed(value):
    """``value`` with the deterministic prefix of every string replaced by this package's (types kept)."""
    if isinstance(value, str):
        return value.replace(DET_PREFIX, ND_PREFIX)
    if isinstance(value, dict):
        return {_renamed(k): _renamed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_renamed(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_renamed(v) for v in value)
    return value


def _strings(value):
    """Every string inside ``value`` (keys included)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _strings(v)


def _dropped(name: str) -> bool:
    return name.startswith(DROPPED)


def _atomic_split(arch: str) -> str:
    """The atomic fn split of ``arch``: the form of the deterministic backward split (``FN_SPLIT_PROGRAM[arch][1]``),
    launched as the token-owner pass's programmatic dependent (``_dep``)."""
    return f"{ND_PREFIX}fn_split_atomic__{det.fn_split_program(arch, 1)}_dep"


def _det_role(name: str) -> str:
    return det.MODULES[name]["role"]


@contextlib.contextmanager
def _torch_deterministic(mode: bool, *, warn_only: bool = False):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


# ----------------------------------------------------------------------------------------------------------------
# The program set and the records


@pytest.mark.cpu_only
def test_program_set_is_the_deterministic_set_without_slab_reduce_and_backward_split() -> (
    None
):
    """``MODULES`` = every deterministic record but ``bwd_slab__*`` / ``bwd_reduce__*`` and the backward fn splits
    under the ``cake_mhc_train_`` prefix (the record equal field for field after the rename: arches, build knobs,
    launch, arguments, rules) plus the three programs of this package; the dropped records are exactly the
    deterministic column-slab and reduction roles, and the backward fn splits are the only other records without a
    copy."""
    dropped = {name for name in det.MODULES if _dropped(name)}
    assert dropped and {_det_role(name) for name in dropped} == set(DROPPED_ROLES)
    assert not [
        name
        for name in det.MODULES
        if name not in dropped and _det_role(name) in DROPPED_ROLES
    ]
    backward_splits = {
        name for name in det.MODULES if _det_role(name) == "fn_split_bwd"
    }
    assert backward_splits and not backward_splits & dropped
    copied = {
        _renamed(name): name
        for name in det.MODULES
        if name not in dropped and name not in backward_splits
    }
    assert all(name.startswith(ND_PREFIX) for name in copied)
    assert not set(copied) & set(ATOMIC)
    expected = set(copied) | set(ATOMIC)
    assert set(nd.MODULES) == expected, sorted(set(nd.MODULES) ^ expected)
    for name, det_name in copied.items():
        assert nd.MODULES[name] == _renamed(det.MODULES[det_name]), name
    for name, record in nd.MODULES.items():
        assert record["kernel_symbol"] == f"kernel_{record['kernel']}", name
        assert record["library"] == record["kernel"], name
        assert record["library"].startswith(ND_PREFIX), name


@pytest.mark.cpu_only
def test_atomic_records_keep_the_deterministic_launch() -> None:
    """The three programs: named after their library and kernel symbol; the fn splits keep the launch and fixed grid
    of the deterministic transposed-layout split of their form but for the static PDL attribute (``use_pdl`` True:
    they follow the token-owner pass and wait for it) and bind ``fn``, ``fnTp``, ``dfn`` and ``accumulate`` (no
    ``fnp``, no ``want_t``); the column slab keeps the launch and grid rule of the deterministic
    ``dynr`` record, binds its arguments without ``dfn_part`` and adds ``dfn``, ``param_partial``, ``dscale``,
    ``dbase``, ``chunks`` and ``accumulate``."""
    for name, arches in ATOMIC.items():
        record = nd.MODULES[name]
        assert tuple(record["arches"]) == arches, name
        assert record["kernel"] == record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["hidden"] == HIDDEN and record["ffi_entry"] == "run", name
        assert record["select_rule"] is None, name
        assert record["workspace_rule"] == {"kind": "regions", "workspace": "bwd"}
        assert name not in nd.RECORD_SCALARS, name
    for arch in det.ARCHES:
        split = nd.MODULES[_atomic_split(arch)]
        assert arch in split["arches"] and split["role"] == "fn_split_bwd", arch
        det_splits = [
            n
            for n in det.route_modules(arch, HIDDEN, "bwd_finish")
            if _det_role(n) == "fn_split_bwd"
        ]
        assert len(det_splits) == 1, arch
        reference = det.MODULES[det_splits[0]]
        assert reference["launch"]["use_pdl"] is False, arch
        assert split["launch"] == {**reference["launch"], "use_pdl": True}, arch
        assert split["grid_rule"] == reference["grid_rule"], arch
        form = det.fn_split_program(arch, 1)
        assert split["grid_rule"] == {"kind": "fixed", "ctas": det.FN_SPLIT_GRID[form]}
        assert set(split["logical"].values()) == {"fn", "fnTp", "dfn", "accumulate"}
        assert [kind for kind, _ in split["arg_plan"]][-3:] == ["grid"] * 3
    slab, dynr = nd.MODULES[SLAB], det.MODULES[DET_DYNR]
    assert slab["role"] == "bwd_slab"
    assert slab["launch"] == dynr["launch"] and slab["launch"]["use_pdl"] == "runtime"
    assert slab["launch"]["block"][0] == 384
    assert slab["grid_rule"] == dynr["grid_rule"]
    added = {"dfn", "param_partial", "dscale", "dbase", "chunks", "accumulate"}
    assert (
        set(slab["logical"].values())
        == (set(dynr["logical"].values()) - {"dfn_part"}) | added
    )
    kept = [entry for entry in dynr["arg_plan"] if entry != ["buffer", "dfn_part"]]
    assert [entry for entry in slab["arg_plan"] if entry[1] not in added] == kept


@pytest.mark.cpu_only
def test_no_record_references_a_deterministic_library() -> None:
    """No string of ``MODULES``, ``ROUTES``, ``RECORD_SCALARS``, ``PDL_EDGES`` or ``PDL_OFF_BANDS`` names the
    deterministic package; every routed name is a record of this package."""
    for table in ("MODULES", "ROUTES", "RECORD_SCALARS", "PDL_EDGES", "PDL_OFF_BANDS"):
        found = [s for s in _strings(getattr(nd, table)) if "det_train" in s]
        assert not found, (table, found[:5])
    for arch, routes in nd.ROUTES.items():
        for key, names in routes.items():
            for name in names:
                assert name in nd.MODULES and name.startswith(ND_PREFIX), (arch, key)
    assert set(nd.RECORD_SCALARS) <= set(nd.MODULES)
    assert all(stem.startswith(ND_PREFIX) for stem in nd.PDL_EDGES)


@pytest.mark.cpu_only
def test_records_speak_the_vocabulary() -> None:
    """Every record binds logical names of this package's vocabulary (``LOGICAL_TENSORS`` = the deterministic one
    without ``dfn_part``; tensors as buffers, scalars as parameters), one logical name per kernel parameter, the
    per-call PDL argument exactly when ``use_pdl`` is ``"runtime"`` (just before the grid).  The host binds every
    argument of every routed record: each logical name is a call value, a workspace region, a host-plan scalar of
    the route's fast plan or a scalar the record fixes (the run-time-count records also take ``sinkhorn_iters``,
    which the full path passes), and every workspace value names a region of the route's workspace."""
    assert list(nd.LOGICAL_TENSORS) == [
        t for t in det.LOGICAL_TENSORS if t != "dfn_part"
    ]
    assert "dfn_part" in det.LOGICAL_TENSORS
    assert nd.LOGICAL_SCALARS == det.LOGICAL_SCALARS
    tensors, scalars = set(nd.LOGICAL_TENSORS), set(nd.LOGICAL_SCALARS)
    for name, record in nd.MODULES.items():
        plan, logical = record["arg_plan"], record["logical"]
        kinds = [kind for kind, _ in plan]
        assert set(kinds) <= {"buffer", "parameter", "launch", "grid"}, name
        assert plan[-3:] == [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
        launches = [i for i, kind in enumerate(kinds) if kind == "launch"]
        if record["launch"]["use_pdl"] == "runtime":
            assert launches == [len(plan) - 4], name
            assert plan[-4] == ["launch", "enable_pdl"], name
        else:
            # a static PDL attribute (``True``) or none: no per-call argument
            assert record["launch"]["use_pdl"] in (True, False), name
            assert not launches, name
        bound = [key for kind, key in plan if kind in ("buffer", "parameter")]
        assert len(bound) == len(set(bound)) and set(bound) == set(logical), name
        for kind, key in plan:
            if kind == "buffer":
                assert logical[key] in tensors, (name, key, logical[key])
            elif kind == "parameter":
                assert logical[key] in scalars, (name, key, logical[key])
    for arch, routes in nd.ROUTES.items():
        for key, names in routes.items():
            kind = key.rsplit("_h", 1)[0]
            assert nd.route_key(HIDDEN, kind) == key
            workspace = nd.WORKSPACE_OF_KIND.get(kind)
            region_names = (
                {r["name"] for r in nd.WORKSPACE_REGIONS[workspace]} | {"dzp_rows"}
                if workspace
                else set()
            )
            assert {region for _, region in pkg._WORKSPACE_VALUES[kind]} <= region_names
            provided = (
                set(pkg._CALL_VALUES[kind])
                | {value for value, _ in pkg._WORKSPACE_VALUES[kind]}
                | set(pkg._HOST_VALUES[kind])
                | {"rows"}
            )
            for name in names:
                record = nd.MODULES[name]
                need = set(record["logical"].values()) - set(
                    nd.RECORD_SCALARS.get(name, {})
                )
                if (record["select_rule"] or {}).get("kind") == "run_time_iters":
                    need.discard("sinkhorn_iters")
                assert need <= provided, (arch, key, name, need - provided)
                rule = record["workspace_rule"]
                if rule is not None:
                    assert rule == {"kind": "regions", "workspace": workspace}, name


# ----------------------------------------------------------------------------------------------------------------
# Routes, selection, PDL


@pytest.mark.cpu_only
def test_routes_are_the_deterministic_routes_but_bwd_finish() -> None:
    """``ROUTES``: the deterministic architectures and route keys; every route the renamed deterministic route except
    ``bwd_finish``, which is the deterministic token-owner records (renamed, in route order), the architecture's
    atomic fn split and the atomic column slab.  Every record is routed, on exactly its architectures."""
    assert set(nd.ROUTES) == set(det.ROUTES) == set(det.ARCHES)
    routed: dict[str, set] = {}
    for arch in det.ARCHES:
        assert set(nd.ROUTES[arch]) == set(det.ROUTES[arch]), arch
        for key, det_route in det.ROUTES[arch].items():
            route = nd.ROUTES[arch][key]
            for name in route:
                routed.setdefault(name, set()).add(arch)
            if key != BWD_FINISH:
                assert route == _renamed(det_route), (arch, key)
                continue
            tokens = [_renamed(n) for n in det_route if _det_role(n) == "bwd_token"]
            assert tokens, arch
            assert route == [*tokens, _atomic_split(arch), SLAB], arch
            assert {_det_role(n) for n in det_route} == {
                "fn_split_bwd",
                "bwd_token",
                *DROPPED_ROLES,
            }, arch
    for name, arches in routed.items():
        assert arches == set(nd.MODULES[name]["arches"]), name
    assert set(nd.MODULES) == set(routed), sorted(set(nd.MODULES) ^ set(routed))


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_selection_is_the_deterministic_selection_but_bwd_finish(
    arch: str, sm: int
) -> None:
    """On every token count of record at the count of record and at run-time counts: every kind selects the renamed
    deterministic records, except ``bwd_finish``: (the deterministic token-owner record(s) of the call renamed, the
    atomic fn split, the atomic column slab) -- one fn split, one column slab, no reduction."""
    for T in T_VALUES:
        for iters in (det.SINKHORN_ITERS, *RUN_TIME_ITERS):
            for kind in det.KINDS:
                what = (arch, sm, T, iters, kind)
                det_names = det.selected_modules(
                    arch, HIDDEN, kind, rows=T, sm_count=sm, sinkhorn_iters=iters
                )
                names = nd.selected_modules(
                    arch, HIDDEN, kind, rows=T, sm_count=sm, sinkhorn_iters=iters
                )
                if kind != "bwd_finish":
                    assert names == tuple(_renamed(n) for n in det_names), what
                    continue
                assert sorted(_det_role(n) for n in det_names) == sorted(
                    ("fn_split_bwd", "bwd_token", *DROPPED_ROLES)
                ), (what, det_names)
                tokens = tuple(
                    _renamed(n) for n in det_names if _det_role(n) == "bwd_token"
                )
                assert names == (*tokens, _atomic_split(arch), SLAB), (what, names)
                run_time = (nd.MODULES[tokens[0]]["select_rule"] or {}).get("kind")
                assert (run_time == "run_time_iters") == (
                    iters != det.SINKHORN_ITERS
                ), what


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_pdl_follows_the_deterministic_edges(arch: str, sm: int) -> None:
    """``PDL_EDGES`` = the deterministic edges without ``K7->K8``, the column slab's stem
    ``cake_mhc_train_bwd_slab_atomic`` on ``K6->K7``; ``PDL_OFF_BANDS`` = the deterministic bands without ``K7->K8``.
    On every path and token count (the counts of record and every band boundary) the column slab's ``enable_pdl``
    equals every deterministic column-slab record's of the route, the fn split takes no per-call argument (its
    attribute is static), and every copied record reads the deterministic value."""
    expected_edges = {
        _renamed(stem): edge
        for stem, edge in det.PDL_EDGES.items()
        if edge != "K7->K8" and not stem.endswith("bwd_slab")
    }
    expected_edges[ND_PREFIX + "bwd_slab_atomic"] = "K6->K7"
    assert expected_edges == nd.PDL_EDGES
    assert "K7->K8" in set(det.PDL_EDGES.values())
    assert {
        a: {edge: bands for edge, bands in edges.items() if edge != "K7->K8"}
        for a, edges in det.PDL_OFF_BANDS.items()
    } == nd.PDL_OFF_BANDS
    assert nd.pdl_edge(nd.MODULES[SLAB]["library"]) == "K6->K7"
    split = _atomic_split(arch)
    det_slabs = [
        n
        for n in det.route_modules(arch, HIDDEN, "bwd_finish")
        if _det_role(n) == "bwd_slab"
    ]
    assert det_slabs
    boundaries = set()
    for bands in det.PDL_OFF_BANDS.get(arch, {}).get("K6->K7", {}).values():
        for above, at_most in bands:
            boundaries |= {above, above + 1}
            if at_most is not None:
                boundaries |= {at_most, at_most + 1}
    copied = [
        n
        for route in det.ROUTES[arch].values()
        for n in route
        if not _dropped(n) and _det_role(n) != "fn_split_bwd"
    ]
    for T in sorted(set(T_VALUES) | {t for t in boundaries if t > 0}):
        for path in det.PDL_PATHS:
            value = nd.enable_pdl(SLAB, T, arch, path)
            assert value in (0, 1)
            for name in det_slabs:
                assert value == det.enable_pdl(name, T, arch, path), (T, path, name)
            assert nd.enable_pdl(split, T, arch, path) == 1, (T, path)
            for name in copied:
                assert nd.enable_pdl(_renamed(name), T, arch, path) == det.enable_pdl(
                    name, T, arch, path
                ), (T, path, name)
    with pytest.raises(ValueError):
        nd.enable_pdl(SLAB, 1, arch, "sideways")


# ----------------------------------------------------------------------------------------------------------------
# Workspaces


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_workspace_is_the_deterministic_layout_without_the_partials(
    arch: str, sm: int
) -> None:
    """``fwd`` is the deterministic forward workspace; the ``bwd`` regions are the deterministic ``bwd`` regions but
    the last (``dfn_part``, ``P x 24 x 16384`` FP32) at identical offsets, sizes, dtypes and shapes, the ticket region
    first; the total is the deterministic total less the rounded ``dfn_part`` bytes; ``T == 0`` has no regions."""
    assert set(nd.WORKSPACE_REGIONS) == set(det.WORKSPACE_REGIONS) == {"fwd", "bwd"}
    assert nd.WORKSPACE_REGIONS["fwd"] == det.WORKSPACE_REGIONS["fwd"]
    assert det.WORKSPACE_REGIONS["bwd"][-1]["name"] == "dfn_part"
    assert nd.WORKSPACE_REGIONS["bwd"] == det.WORKSPACE_REGIONS["bwd"][:-1]
    assert nd.WORKSPACE_OF_KIND == det.WORKSPACE_OF_KIND
    for T in det.RULES_T_VALUES:
        assert nd.workspace_regions("fwd", T, sm, arch) == det.workspace_regions(
            "fwd", T, sm, arch
        ), T
        layout = nd.workspace_regions("bwd", T, sm, arch)
        reference = det.workspace_regions("bwd", T, sm, arch)
        assert layout["total_bytes"] == nd.workspace_bytes("bwd", T, sm, arch), T
        if T == 0:
            assert layout == {"regions": {}, "total_bytes": 0}
            continue
        names = list(reference["regions"])
        assert names[-1] == "dfn_part" and list(layout["regions"]) == names[:-1], T
        for name, region in layout["regions"].items():
            assert region == reference["regions"][name], (T, name)
        assert layout["regions"]["ticket"]["offset"] == 0
        assert layout["regions"]["ticket"]["bytes"] == 4 * det.TICKET_WORDS
        part = reference["regions"]["dfn_part"]
        parts = det.k7_chunks(T, sm)[0]
        assert part["shape"] == (parts, 24, 16384), T
        assert part["bytes"] == parts * 24 * 16384 * 4, T
        assert layout["total_bytes"] == reference["total_bytes"] - _round256(
            part["bytes"]
        ), T
        assert layout["total_bytes"] % 256 == 0, T
    with pytest.raises(ValueError):
        nd.workspace_regions("bwd", -1, sm, arch)
    with pytest.raises(ValueError):
        nd.workspace_regions("dfn", 1, sm, arch)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_workspace_envelope_covers_every_smaller_token_count(
    arch: str, sm: int
) -> None:
    """``workspace_bytes_upto("bwd", T)`` is non-decreasing, at least the exact size of every token count up to ``T``
    and at most the deterministic envelope (a buffer of the deterministic sizing serves this package); 0 at
    ``T == 0``; the forward envelope is the deterministic one."""
    previous = 0
    for T in range(0, 4200):
        envelope = nd.workspace_bytes_upto("bwd", T, sm, arch)
        assert envelope >= previous, T
        assert envelope >= nd.workspace_bytes("bwd", T, sm, arch), T
        assert envelope <= det.workspace_bytes_upto("bwd", T, sm, arch), T
        previous = envelope
    assert nd.workspace_bytes_upto("bwd", 0, sm, arch) == 0
    for rows in (64, 65, 1000, 2049, 8192, 16263, 20001, 65536):
        envelope = nd.workspace_bytes_upto("bwd", rows, sm, arch)
        assert envelope <= det.workspace_bytes_upto("bwd", rows, sm, arch), rows
        assert nd.workspace_bytes_upto(
            "fwd", rows, sm, arch
        ) == det.workspace_bytes_upto("fwd", rows, sm, arch), rows
        for T in sorted(
            {t for t in det.RULES_T_VALUES if t <= rows}
            | {rows, rows - 1, (rows + 1) // 2}
        ):
            assert envelope >= nd.workspace_bytes("bwd", T, sm, arch), (rows, T)


# ----------------------------------------------------------------------------------------------------------------
# Host rules and route glue


# The constants and tables of the host rules (the deterministic package's values, copied).
_SHARED_TABLES = (
    "ARCHES", "ARCH_BY_CAPABILITY", "KINDS", "STREAMS", "HIDDEN", "HIDDEN_TOTAL", "MAPPING_ROWS", "N_PAD", "PIECES",
    "SLAB_D", "TILE_T", "SLAB_TILE_T", "RMS_EPS", "HC_EPS", "SINKHORN_ITERS", "SINKHORN_ITERS_MAX", "STATS_COLS",
    "GSTATS_COLS", "TOKP_COLS", "DZP_COLS", "PARAM_COLS", "TICKET_WORDS", "STATS_LAYOUT", "GSTATS_LAYOUT",
    "TOKP_LAYOUT", "PARAM_LAYOUT", "RULES_SM_COUNTS", "RULES_BOARDS", "RULES_T_VALUES", "PK_BANDS",
    "FN_SPLIT_PROGRAM", "FN_SPLIT_GRID", "FWD_MAP_PRODUCTION", "FWD_MAP_BANDS", "FWD_MAP_FIT", "FWD_MAP_CLAIM",
    "FWD_COEF_REGIMES", "BWD_TOKEN_PROGRAM", "BWD_TOKEN_REGIMES", "BWD_TOKEN_BANDS", "K5_GRID_BANDS",
    "K5_FORM_BANDS", "K5_FORM_ROWS", "FWD_FINISH_PROGRAM", "FWD_FINISH_ROWS", "HIDDEN_SIZES", "PDL_PATHS",
    "SELECT_RULE_KINDS",
)  # fmt: skip
# The deterministic column-slab and reduction rules, absent here.
_DROPPED_RULES = (
    "BWD_SLAB_PROGRAM", "BWD_SLAB_BANDS", "BWD_SLAB_FIT", "BWD_REDUCE_PROGRAM", "BWD_REDUCE_REGIMES",
    "BWD_REDUCE_PARTS", "BWD_REDUCE_GRID", "bwd_slab_program", "bwd_reduce_program", "bwd_reduce_grid",
)  # fmt: skip


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_host_rules_are_copies_of_the_deterministic_ones(arch: str, sm: int) -> None:
    """The constants and tables of the host rules equal the deterministic ones; the rule functions are this
    package's own objects with the deterministic outputs on every token count of record; the deterministic
    column-slab and reduction rules are absent and ``PROGRAM_RULES`` drops them; a rules row is the deterministic row
    without its column-slab and reduction program columns, with this package's backward workspace bytes."""
    for name in _SHARED_TABLES:
        assert getattr(nd, name) == getattr(det, name), name
    for name in _DROPPED_RULES:
        assert hasattr(det, name) and not hasattr(nd, name), name
    assert (
        tuple(rule for rule in det.PROGRAM_RULES if rule not in _DROPPED_RULES)
        == nd.PROGRAM_RULES
    )
    for name in (
        "host_plan",
        "k1_units",
        "k7_chunks",
        "k6_grid",
        "t_pad",
        "selected_modules",
        "enable_pdl",
    ):
        assert getattr(nd, name) is not getattr(det, name), name
    for layout in (0, 1):
        assert nd.fn_split_program(arch, layout) == det.fn_split_program(arch, layout)
        assert nd.fn_split_grid(arch, layout) == det.fn_split_grid(arch, layout)
    assert nd.fwd_finish_program(arch) == det.fwd_finish_program(arch)
    dropped_columns = {"bwd_slab_program", "bwd_reduce_program", "bwd_reduce_grid"}
    for T in det.RULES_T_VALUES:
        row, reference = nd.rules_row(T, sm, arch), det.rules_row(T, sm, arch)
        assert dropped_columns <= set(reference) and not dropped_columns & set(row), T
        expected = {k: v for k, v in reference.items() if k not in dropped_columns}
        expected["bwd_workspace_bytes"] = nd.workspace_bytes("bwd", T, sm, arch)
        assert row == expected, T
        if T == 0:
            continue
        assert nd.host_plan(T, sm, arch) == det.host_plan(T, sm, arch), T
        for rule in (
            "fwd_map_program",
            "fwd_coef_program",
            "bwd_token_program",
            "k5_program",
            "k5_grid",
        ):
            assert getattr(nd, rule)(T, sm, arch) == getattr(det, rule)(T, sm, arch), (
                rule,
                T,
            )
        assert nd.k7_chunks(T, sm) == det.k7_chunks(T, sm), T
        assert nd.k6_grid(T, sm) == det.k6_grid(T, sm), T


@pytest.mark.cpu_only
def test_route_glue_and_supported_hidden_sizes() -> None:
    for arch in det.ARCHES:
        assert (
            nd.supported_hidden_sizes(arch)
            == det.supported_hidden_sizes(arch)
            == (HIDDEN,)
        ), arch
    assert nd.supported_hidden_sizes("sm_90a") == ()
    for kind in det.KINDS:
        assert nd.route_key(HIDDEN, kind) == det.route_key(HIDDEN, kind)
        for capability in ((10, 0), (10, 3), (10, 7)):
            assert nd.route_applies(
                device_capability=capability, hidden=HIDDEN, kind=kind
            )
            assert not nd.route_applies(
                device_capability=capability, hidden=2048, kind=kind
            )
        for capability in ((9, 0), (12, 0), (10, 1)):
            assert not nd.route_applies(
                device_capability=capability, hidden=HIDDEN, kind=kind
            )
    with pytest.raises(ValueError):
        nd.route_key(HIDDEN, "bwd_reduce")
    with pytest.raises(ValueError):
        nd.selected_modules("sm_90a", HIDDEN, "bwd_finish", rows=129, sm_count=148)
    with pytest.raises(ValueError):
        nd.selected_modules("sm_100a", 2048, "bwd_finish", rows=129, sm_count=148)
    assert pkg.route_modules is nd.route_modules and pkg.route_key is nd.route_key


@pytest.mark.cpu_only
def test_library_names_and_record_scalars() -> None:
    """``library_name`` is ``<library>_<arch>`` on the record's architectures and a ``RuntimeError`` elsewhere;
    ``RECORD_SCALARS`` is the deterministic table renamed without the backward fn splits (the forward fn split
    ``want_t = 0``; the atomic fn splits take no ``want_t``)."""
    for name, record in nd.MODULES.items():
        for arch in det.ARCHES:
            if arch in record["arches"]:
                assert nd.library_name(name, arch) == f"{record['library']}_{arch}"
            else:
                with pytest.raises(RuntimeError, match=f"not delivered for {arch}"):
                    nd.library_name(name, arch)
    assert {
        _renamed(name): scalars
        for name, scalars in det.RECORD_SCALARS.items()
        if _det_role(name) != "fn_split_bwd"
    } == nd.RECORD_SCALARS
    assert nd.RECORD_SCALARS == {ND_PREFIX + "fn_split": {"want_t": 0}}


# ----------------------------------------------------------------------------------------------------------------
# The package is standalone


def _imports(path: Path) -> list[tuple[ast.AST, bool]]:
    """Every import statement of ``path`` with whether it sits inside a function body."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[ast.AST, bool]] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append((child, in_function))
            visit(
                child,
                in_function
                or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
            )

    visit(tree, False)
    return found


@pytest.mark.cpu_only
def test_package_imports_nothing_from_other_flashinfer_packages() -> None:
    """No source file of the package imports from a parent package (``from ..``) or another flashinfer module; a
    relative import names a module of the package itself; nothing imports by string.  The one exception is the
    shared loader ``cake_build.py`` (byte-identical to the deterministic package's), whose JIT fallback imports
    ``flashinfer.jit`` inside a function when no prebuilt library exists."""
    sources = sorted(PACKAGE_DIR.glob("*.py"))
    assert {p.name for p in sources} == {"__init__.py", "cake_jit.py", "cake_build.py"}
    assert (PACKAGE_DIR / "cake_build.py").read_bytes() == Path(
        det_build.__file__
    ).read_bytes()
    jit_fallback = []
    for path in sources:
        text = path.read_text()
        assert "import_module(" not in text and "__import__(" not in text, path.name
        for node, in_function in _imports(path):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (path.name, node.lineno, node.level)
                target = node.module.split(".")[0] if node.module else None
                names = [target] if target else [alias.name for alias in node.names]
                for name in names:
                    assert (PACKAGE_DIR / f"{name}.py").is_file(), (path.name, name)
                continue
            modules = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
            )
            for module in modules:
                if module == "flashinfer" or module.startswith("flashinfer."):
                    assert path.name == "cake_build.py" and in_function, (
                        path.name,
                        node.lineno,
                        module,
                    )
                    assert module.startswith("flashinfer.jit."), (path.name, module)
                    jit_fallback.append(module)
    assert jit_fallback, "the cake_build JIT fallback import was not found"


_VENDORED_CHILD = r"""
import importlib, importlib.abc, json, sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError("flashinfer is blocked: " + name)
        return None


sys.meta_path.insert(0, Block())
sys.modules["flashinfer"] = None
sys.path.insert(0, sys.argv[1])
import torch

package = importlib.import_module(sys.argv[2])
jit = importlib.import_module(sys.argv[2] + ".cake_jit")
importlib.import_module(sys.argv[2] + ".cake_build")
out = {
    "supported_cpu": package.is_cake_mhc_train_supported(torch.device("cpu")),
    "selection": {
        f"{arch}/{sm}": list(jit.selected_modules(arch, 4096, "bwd_finish", rows=4096, sm_count=sm))
        for arch, sm in jit.RULES_BOARDS
    },
    "bwd_bytes": jit.workspace_bytes("bwd", 4096, 148, "sm_100a"),
}
try:
    package.cake_mhc_train_backward_finish(*([None] * 9))
except ValueError as error:
    out["refusal"] = str(error)
out["flashinfer"] = sorted(
    name for name, module in sys.modules.items()
    if module is not None and (name == "flashinfer" or name.startswith("flashinfer."))
)
print(json.dumps(out))
"""


@pytest.mark.cpu_only
def test_vendored_copy_imports_with_flashinfer_blocked(tmp_path: Path) -> None:
    """The package directory copied under another top-level name (the vendored gate's first step, without its
    sources) imports, plans its routes and refuses ``deterministic=True`` in an interpreter where every
    ``flashinfer`` import raises; no flashinfer module is loaded."""
    neutral = "vendored_cake_mhc_train_backend"
    shutil.copytree(
        PACKAGE_DIR,
        tmp_path / neutral,
        ignore=shutil.ignore_patterns("aot", "__pycache__", "csrc", "build.json"),
    )
    run = subprocess.run(
        [sys.executable, "-c", _VENDORED_CHILD, str(tmp_path), neutral],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-8:])
    assert run.returncode == 0, tail
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["flashinfer"] == [], out["flashinfer"]
    assert out["supported_cpu"] is False
    assert len(out["selection"]) == len(det.RULES_BOARDS)
    for arch, sm in det.RULES_BOARDS:
        assert out["selection"][f"{arch}/{sm}"] == list(
            nd.selected_modules(arch, HIDDEN, "bwd_finish", rows=4096, sm_count=sm)
        ), (arch, sm)
    assert out["bwd_bytes"] == nd.workspace_bytes("bwd", 4096, 148, "sm_100a")
    assert out["refusal"].startswith(f"cake_mhc_train_backward_finish {_REFUSAL}")


# ----------------------------------------------------------------------------------------------------------------
# The mode policy (no device: the refusal and the alert come before any device work)


@pytest.mark.cpu_only
def test_exports() -> None:
    """The package exports the det-free names only (no ``cake_mhc_det_train_*`` / ``CakeMHCDetTrain*`` name, no head
    entry point) and every exported name exists."""
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
        assert "det_train" not in name and "DetTrain" not in name, name
        assert "head" not in name.lower(), name
    assert set(MODE_ENTRIES) | set(MODELESS) <= set(pkg.__all__)
    assert {"CakeMHCTrainPrepareFunction", "CakeMHCTrainFinishFunction"} <= set(
        pkg.__all__
    )
    assert pkg.SUPPORTED_COMPUTE_CAPABILITIES == ((10, 0), (10, 3), (10, 7))
    assert pkg.__name__ == PACKAGE_NAME


@pytest.mark.cpu_only
def test_entry_points_take_the_deterministic_keyword() -> None:
    for name in MODE_ENTRIES:
        parameter = inspect.signature(getattr(pkg, name)).parameters["deterministic"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is True, name
    for name in MODELESS:
        assert (
            "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
        ), name


def _cpu_calls(**mode) -> dict:
    """One call per mode entry point on CPU operands of 8 tokens (``mode``: the keyword, ``{}`` the default)."""
    T, cpu = 8, torch.device("cpu")
    X = torch.zeros(T, 4, 4096, dtype=torch.bfloat16)
    fn = torch.zeros(24, 16384)
    scale, base = torch.ones(3), torch.zeros(24)
    B, dU = (
        torch.zeros(T, 4096, dtype=torch.bfloat16),
        torch.zeros(T, 4096, dtype=torch.bfloat16),
    )
    stats, gstats = torch.zeros(T, 64), torch.zeros(T, 32)
    return {
        "cake_mhc_train_forward_prepare": lambda: pkg.cake_mhc_train_forward_prepare(X, fn, scale, base, **mode),
        "cake_mhc_train_forward_stats": lambda: pkg.cake_mhc_train_forward_stats(X, fn, scale, base, **mode),
        "cake_mhc_train_forward_finish": lambda: pkg.cake_mhc_train_forward_finish(X, B, stats, **mode),
        "cake_mhc_train_backward_prepare": lambda: pkg.cake_mhc_train_backward_prepare(X, B, X, stats, **mode),
        "cake_mhc_train_backward_finish": lambda: pkg.cake_mhc_train_backward_finish(
            X, X, dU, B, fn, scale, base, stats, gstats, **mode
        ),
        "cake_mhc_train_prepare": lambda: pkg.cake_mhc_train_prepare(X, fn, scale, base, **mode),
        "cake_mhc_train_finish": lambda: pkg.cake_mhc_train_finish(X, B, object(), **mode),
        "cake_mhc_train_forward_workspace_bytes": lambda: pkg.cake_mhc_train_forward_workspace_bytes(T, cpu, **mode),
        "cake_mhc_train_backward_workspace_bytes": lambda: pkg.cake_mhc_train_backward_workspace_bytes(T, cpu, **mode),
        "cake_mhc_train_forward_workspace": lambda: pkg.cake_mhc_train_forward_workspace(T, cpu, **mode),
        "cake_mhc_train_backward_workspace": lambda: pkg.cake_mhc_train_backward_workspace(T, cpu, **mode),
    }  # fmt: skip


_MODES = (
    {},
    {"deterministic": True},
    {"deterministic": None},
    {"deterministic": 0},
    {"deterministic": 1},
    {"deterministic": "False"},
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mode", _MODES, ids=["default", "true", "none", "zero", "one", "string"]
)
def test_entry_points_refuse_a_deterministic_mode_first(mode: dict) -> None:
    """Any ``deterministic`` but the bool ``False`` (the default ``True`` included) is a ``ValueError`` naming the
    entry point and the deterministic package, before any operand, device or route check -- also with PyTorch's
    deterministic algorithms on (the refusal precedes the alert)."""
    calls = _cpu_calls(**mode)
    assert set(calls) == set(MODE_ENTRIES)
    for torch_mode in ((False, False), (True, False), (True, True)):
        with _torch_deterministic(torch_mode[0], warn_only=torch_mode[1]):
            for name, call in calls.items():
                with pytest.raises(ValueError, match=f"^{name} {_REFUSAL}") as info:
                    call()
                assert (
                    "flashinfer.cake_mhc_det_train is the deterministic package"
                    in str(info.value)
                ), name


def _expected_alert(entry: str, warn_only: bool) -> str:
    if warn_only:
        return (
            f"{entry} {_ALERT}, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'; "
            "flashinfer.cake_mhc_det_train is the deterministic package"
        )
    return (
        f"{entry} {_ALERT}, but you set 'torch.use_deterministic_algorithms(True)'; "
        "flashinfer.cake_mhc_det_train is the deterministic package, or use the 'warn_only=True' option if that is "
        "acceptable for your application"
    )


@pytest.mark.cpu_only
def test_alert_wording_in_both_torch_modes() -> None:
    """The package's alert (no private PyTorch hook): silent with PyTorch's deterministic algorithms off, a
    ``RuntimeError`` with the shared wording when on, a ``UserWarning`` with it under ``warn_only=True``."""
    with _torch_deterministic(False), warnings.catch_warnings():
        warnings.simplefilter("error")
        pkg._alert_nondeterministic("probe_entry")
    with _torch_deterministic(True):
        with pytest.raises(RuntimeError) as info:
            pkg._alert_nondeterministic("probe_entry")
        assert str(info.value) == _expected_alert("probe_entry", False)
    with _torch_deterministic(True, warn_only=True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pkg._alert_nondeterministic("probe_entry")
        assert [(w.category, str(w.message)) for w in caught] == [
            (UserWarning, _expected_alert("probe_entry", True))
        ]
    assert "_alert_not_deterministic" not in Path(pkg.__file__).read_text()


@pytest.mark.cpu_only
def test_no_entry_alerts_before_its_operands_pass(monkeypatch) -> None:
    """With ``deterministic=False`` under PyTorch's deterministic algorithms (strict and ``warn_only=True``): the alert
    belongs to the unordered ``dfn`` reduction, so it is raised right before the launches of ``backward_finish``
    (validated operands, ``T > 0``; the GPU suite checks it there and that ``T == 0`` does not alert).  On the CPU
    operands here every entry point, ``backward_finish`` included, refuses its operands with no alert and launches
    nothing (warnings are errors)."""
    launched = []
    monkeypatch.setattr(pkg, "_launch_route", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(pkg, "_run_route", lambda *a, **k: launched.append(a))
    calls = _cpu_calls(deterministic=False)
    assert "cake_mhc_train_backward_finish" in calls
    for warn_only in (False, True):
        with _torch_deterministic(True, warn_only=warn_only), warnings.catch_warnings():
            warnings.simplefilter("error")
            for name, call in calls.items():
                with pytest.raises((ValueError, TypeError)) as info:
                    call()
                assert _ALERT not in str(info.value), (name, warn_only)
                assert _REFUSAL not in str(info.value), (name, warn_only)
    assert launched == []


# ----------------------------------------------------------------------------------------------------------------
# Wiring and the build manifest


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    """The test-helper key, ``flashinfer.experimental.cake`` (a docstring only), the package data, the ignored
    ``aot/`` directory and the pre-commit exclude of the generated sources; the legacy backend directory keeps its
    alias module only (no registry, manifest or sources)."""
    assert PACKAGES[KEY] == PACKAGE_NAME == pkg.__name__
    assert "flashinfer.cake_mhc_train_backend" not in PACKAGES.values()
    assert "cake_mhc_train" not in PACKAGES
    tree = ast.parse((REPO / "flashinfer/experimental/cake/__init__.py").read_text())
    assert ast.get_docstring(tree)
    assert len(tree.body) == 1, "flashinfer.experimental.cake holds a docstring only"
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    assert package_data[PACKAGE_NAME] == ["csrc/**", "build.json"]
    assert "flashinfer.cake_mhc_train_backend" not in package_data
    assert "aot/" in (PACKAGE_DIR / ".gitignore").read_text().split()
    assert (
        "experimental/cake/cake_mhc_train_backend/csrc/"
        in (REPO / ".pre-commit-config.yaml").read_text()
    )
    legacy = REPO / "flashinfer" / "cake_mhc_train_backend"
    assert sorted(p.name for p in legacy.iterdir() if p.name != "__pycache__") == [
        "__init__.py"
    ]


@pytest.mark.cpu_only
def test_build_manifest_lists_every_program() -> None:
    """``build.json`` (schema ``cake-build/1``) names this package and carries exactly one entry per (record library,
    architecture), each with its kernel and binding sources under ``csrc/`` and their digests, IEEE division and no
    fast math; the sources are named after the libraries of this package only."""
    assert (PACKAGE_DIR / "build.json").is_file(), (
        "the package is not rendered (no build.json)"
    )
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_NAME
    libraries = cake_build.libraries_by_name()
    expected = {
        nd.library_name(name, arch)
        for name, record in nd.MODULES.items()
        for arch in record["arches"]
    }
    assert set(libraries) == expected, sorted(set(libraries) ^ expected)
    for name, record in nd.MODULES.items():
        for arch in record["arches"]:
            entry = libraries[nd.library_name(name, arch)]
            assert entry["arch"] == arch and entry["entry"] == "run", (name, arch)
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
            for source in entry["sources"]:
                assert (PACKAGE_DIR / source).is_file(), source
            assert set(entry["sha256"]) == set(entry["sources"])
    stems = {record["library"] for record in nd.MODULES.values()}
    for path in (PACKAGE_DIR / "csrc").glob("*.cu"):
        stem = path.name.rsplit("_", 1)[0]
        assert stem in stems and stem.startswith(ND_PREFIX), path.name


def _manifest_checker():
    """``scripts/check_cake_build_manifests.py`` of this checkout, loaded as a module (it never imports flashinfer)."""
    path = REPO / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
def test_manifest_checker_lists_the_package() -> None:
    """The build-manifest checker lists this package once (generic registry walker over ``cake_jit.py``) and no legacy
    ``flashinfer.cake_mhc_train_backend``; its vendored gate finds the package's suite key; the in-tree checks report
    nothing."""
    checker = _manifest_checker()
    try:
        path = str(PACKAGE_DIR.relative_to(REPO.resolve()))
    except ValueError:
        pytest.skip(
            f"the package is imported from {PACKAGE_DIR}, outside this checkout"
        )
    entries = [p for p in checker.PACKAGES if p.name == PACKAGE_NAME]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert (entry.path, entry.loader) == (path, "cake_jit.py")
    assert entry.refs is checker.refs_generic
    assert not [
        p for p in checker.PACKAGES if p.name == "flashinfer.cake_mhc_train_backend"
    ]
    assert checker.suite_key(REPO, entry) == KEY
    assert checker.run_in_tree(REPO, [entry]) == []
