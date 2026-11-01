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
prefix at its hidden sizes (4096 on every architecture, 5120 and 7168 on
sm_100a / sm_103a), except the ``bwd_finish`` routes.  This module checks the
registry ``cake_jit.py`` against the deterministic registry's records, routes
and record scalars of every hidden size on its boards of record
(``RULES_BOARDS``) and token counts of record (``RULES_T_VALUES``):

* the program set is the deterministic set without its column-slab
  (``bwd_slab__*``) and final-reduction (``bwd_reduce__*``) records, renamed
  and otherwise equal field for field, plus the programs of this package: at
  hidden 4096 ``cake_mhc_train_fn_split_atomic__tv128_dep`` on sm_100a /
  sm_103a, ``cake_mhc_train_fn_split_atomic__tv128_t128_dep`` on sm_107a,
  ``cake_mhc_train_bwd_slab_atomic__dynr`` and
  ``cake_mhc_train_bwd_slab_atomic__lpt`` on all three and their slab-fastest
  copies ``cake_mhc_train_bwd_slab_atomic__dynr_os`` /
  ``cake_mhc_train_bwd_slab_atomic__lpt_os`` on sm_100a / sm_103a; at 5120
  and 7168 ``cake_mhc_train_fn_split_atomic__tv128_dep__h<H>``,
  ``cake_mhc_train_bwd_slab_atomic__dynr__h<H>`` and
  ``cake_mhc_train_bwd_slab_atomic__dynr_os__h<H>`` on sm_100a / sm_103a and
  ``cake_mhc_train_bwd_slab_atomic__lpt__h<H>`` /
  ``cake_mhc_train_bwd_slab_atomic__lpt_os__h<H>`` on sm_100a.  Their
  records keep the launch, grid and roles of the deterministic records they
  replace (the fn splits with the static PDL attribute: they follow the
  token-owner pass), and each ``_os`` record is its base record of the same
  size under its own name and program, on the architectures of its base record
  whose ``BWD_SLAB_ORDER`` of the size has a ``"slab"`` band (none on sm_107a);
* no record, route, record scalar or PDL edge names a deterministic library;
* every route, the selection on every route and the PDL bands equal the
  deterministic ones, except ``bwd_finish``: the deterministic token-owner
  records, the atomic fn split of the architecture and size, the column
  slab's forms (one per call by ``bwd_slab_atomic_program``: ``dynr``,
  ``lpt``, ``dynr_os`` and ``lpt_os`` at hidden 4096 on sm_100a / sm_103a and
  at 5120 and 7168 on sm_100a; ``dynr`` and ``lpt`` on sm_107a, whose unit
  order is ``"chunk"`` at every token count; on sm_103a at 5120 and 7168
  ``dynr`` and ``dynr_os``), no reduction; every program the column slab's
  tables can name has a record on the route, and every ``_os`` record a band
  of the tables that names it;
* the workspaces: ``fwd`` the deterministic one, ``bwd`` the deterministic
  regions at the same offsets without the trailing ``dfn_part``, at every
  hidden size;
* the host rules are copies (equal outputs, not the deterministic objects;
  the per-size tables the deterministic ones without the column-slab and
  reduction tables), but this package's own column-slab rule
  (``BWD_SLAB_LPT``, ``BWD_SLAB_LPT_HIDDEN``, ``BWD_SLAB_ORDER``,
  ``lpt_chunks``, ``bwd_slab_order``, ``bwd_slab_atomic_program``,
  ``slab_plan``; ``host_plan`` itself is the deterministic one): its
  geometries, its token-count bands (the unit order of record, and the banded
  calibration form under a test table) and its selection at every hidden
  size (where the size's table calibrates the architecture: every
  architecture at 4096, sm_100a at 5120 and 7168; elsewhere the one-size
  chunks of ``k7_chunks`` without running the screen), its call-time lookup
  and a workspace envelope that never runs its screen;
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
import heapq
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
# The package's hidden sizes (the deterministic package's): 4096 on every architecture, 5120 and 7168 on the
# architectures of NEW_HIDDEN_ARCHES.
HIDDEN_SIZES = (4096, 5120, 7168)
NEW_HIDDEN_SIZES = HIDDEN_SIZES[1:]
NEW_HIDDEN_ARCHES = ("sm_100a", "sm_103a")
# The architectures of the two-size column slab's records at 5120 and 7168 (the sizes' lpt calibrations).
NEW_HIDDEN_LPT_ARCHES = ("sm_100a",)
ND_PREFIX, DET_PREFIX = "cake_mhc_train_", "cake_mhc_det_train_"
# The programs of this package and their architectures (the library suffix ``__h<H>`` names a size other than 4096).
ATOMIC = {
    "cake_mhc_train_fn_split_atomic__tv128_dep": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fn_split_atomic__tv128_t128_dep": ("sm_107a",),
    "cake_mhc_train_bwd_slab_atomic__dynr": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_slab_atomic__lpt": ("sm_100a", "sm_103a", "sm_107a"),
    **{
        f"cake_mhc_train_{program}__h{hidden}": NEW_HIDDEN_ARCHES
        for hidden in NEW_HIDDEN_SIZES
        for program in ("fn_split_atomic__tv128_dep", "bwd_slab_atomic__dynr")
    },
    **{
        f"cake_mhc_train_bwd_slab_atomic__lpt__h{hidden}": NEW_HIDDEN_LPT_ARCHES
        for hidden in NEW_HIDDEN_SIZES
    },
    # The slab-fastest copies, on the architectures whose BWD_SLAB_ORDER bands of the size hold a "slab" band
    # (lpt_os where the size's two-size calibration of the architecture is not None too).
    "cake_mhc_train_bwd_slab_atomic__dynr_os": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab_atomic__lpt_os": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab_atomic__dynr_os__h5120": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab_atomic__lpt_os__h5120": ("sm_100a",),
    "cake_mhc_train_bwd_slab_atomic__dynr_os__h7168": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab_atomic__lpt_os__h7168": ("sm_100a",),
}
SLAB = "cake_mhc_train_bwd_slab_atomic__dynr"
LPT = "cake_mhc_train_bwd_slab_atomic__lpt"
# The slab-fastest copies of the column slab's two forms (the units drawn slab-fastest; program key ``<form>_os``).
SLAB_OS = "cake_mhc_train_bwd_slab_atomic__dynr_os"
LPT_OS = "cake_mhc_train_bwd_slab_atomic__lpt_os"
DET_DYNR = "cake_mhc_det_train_bwd_slab__dynr"
# The column slab's tables of record: the unit order per hidden size and architecture (BWD_SLAB_ORDER: on sm_100a
# "chunk" to 2047 / 5108 tokens at hidden 4096 / 7168 and "slab" above, "slab" at every token count at 5120; on
# sm_103a "chunk" to 13278 / 28825 / 63091 tokens at hidden 4096 / 5120 / 7168 and "slab" above; "chunk" at every size
# on sm_107a, which routes neither 5120 nor 7168) and the two-size calibration at hidden 4096 (BWD_SLAB_LPT: one
# (o_tiles, keep) per architecture).
ORDER = {
    4096: {
        "sm_100a": ((2047, "chunk"), (None, "slab")),
        "sm_103a": ((13278, "chunk"), (None, "slab")),
        "sm_107a": ((None, "chunk"),),
    },
    5120: {
        "sm_100a": ((None, "slab"),),
        "sm_103a": ((28825, "chunk"), (None, "slab")),
        "sm_107a": ((None, "chunk"),),
    },
    7168: {
        "sm_100a": ((5108, "chunk"), (None, "slab")),
        "sm_103a": ((63091, "chunk"), (None, "slab")),
        "sm_107a": ((None, "chunk"),),
    },
}
LPT_OF_RECORD = {"sm_100a": (0.5, 0.99), "sm_103a": (0.5, 0.98), "sm_107a": (0.5, 0.97)}
# The lpt record's chunk scalars (cake_jit.slab_plan), in kernel-parameter order (P, rpc, p_big, rpc_small).
LPT_HOST_KEYS = (
    "lpt_parts",
    "lpt_rows_per_chunk",
    "lpt_big_parts",
    "lpt_small_rows_per_chunk",
)
# Token counts the two-size rule is checked at beside those of record: the first and the last token of every 37th
# 64-token tile from the first (tiles 1, 38, ..., 1000: up to 64000 tokens).
LPT_SWEEP = tuple(
    sorted({t for k in range(1, 1025, 37) for t in (64 * k - 63, 64 * k)})
)
# The kernel test's screen calibration (o_tiles, keep), the working value the rule is exercised at on every board.
LPT_SCREEN = (1.0, 0.99)
# The 19 benchmark token counts of the BWD_SLAB_LPT calibration, and per board the geometries the table of record picks
# there (the token counts its comment names; every other one keeps the one-size form: dynr, or dynr_os where the
# BWD_SLAB_ORDER band is "slab").
BENCH_T = (1024, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 8192, 14928, 15288, 15566, 15590, 16114, 16132, 16164,
           16263, 32768, 65536)  # fmt: skip
LPT_FIRES = (
    ("sm_100a", 148, {1536: (3, 704, 2, 128), 2048: (3, 960, 2, 128), 2560: (3, 1152, 2, 256), 3584: (4, 1600, 2, 192),
                      14928: (12, 1536, 9, 384), 15566: (12, 1600, 9, 448), 15590: (12, 1600, 9, 448),
                      16114: (12, 1664, 9, 384)}),
    ("sm_103a", 152, {1024: (3, 448, 2, 128), 1536: (3, 704, 2, 128), 2048: (9, 256, 7, 128), 8192: (10, 1088, 7, 192),
                      14928: (8, 2112, 7, 192), 15288: (10, 2048, 7, 320), 16132: (10, 2176, 7, 320),
                      16164: (10, 2176, 7, 320), 16263: (8, 2304, 7, 192)}),
    ("sm_107a", 212, {1024: (4, 320, 3, 64), 2048: (4, 640, 3, 128), 2560: (4, 832, 3, 64), 3072: (4, 960, 3, 192),
                      3584: (4, 1152, 3, 128)}),
)  # fmt: skip
# At hidden 5120 and 7168, the token counts the two-size rule of the size is pinned at (the benchmark token counts and
# 24576), and per board and size the geometries the table of record (BWD_SLAB_LPT_HIDDEN) picks there: on B200 the
# token counts its comment names (H 5120 1024 / 8192 / 15566 / 15590 / 16263 / 32768; H 7168 1024 / 2048 / 3072 /
# 15288 / 24576); nowhere on the boards without a calibration of the size (sm_103a, sm_107a).
LPT_HIDDEN_T = tuple(sorted({*BENCH_T, 24576}))
LPT_FIRES_HIDDEN = (
    ("sm_100a", 148, 5120, {1024: (9, 128, 7, 64), 8192: (13, 704, 11, 256), 15566: (13, 1408, 11, 64),
                            15590: (13, 1408, 11, 64), 16263: (13, 1472, 11, 64), 32768: (13, 2816, 11, 896)}),
    ("sm_100a", 148, 7168, {1024: (2, 896, 1, 128), 2048: (7, 384, 5, 64), 3072: (7, 576, 5, 128),
                            15288: (10, 1664, 9, 320), 24576: (11, 2560, 9, 768)}),
    *((arch, sm, hidden, {}) for arch, sm in det.RULES_BOARDS if arch != "sm_100a" for hidden in NEW_HIDDEN_SIZES),
)  # fmt: skip
# The deterministic programs this package does not copy: the column slab and the final reduction (the backward fn
# splits, whose libraries the forward does not share, have no copy either).
DROPPED = (DET_PREFIX + "bwd_slab__", DET_PREFIX + "bwd_reduce__")
DROPPED_ROLES = ("bwd_slab", "bwd_reduce")
# Sinkhorn counts other than the count of record: the token-owner pass runs its run-time-count record.
RUN_TIME_ITERS = (1, 7, 64)
T_VALUES = tuple(T for T in det.RULES_T_VALUES if T > 0)
# The deterministic records, routes and record scalars of this package's hidden sizes (every size the deterministic
# package exports: this package copies each of them).
DET_MODULES = {
    name: record
    for name, record in det.MODULES.items()
    if record["hidden"] in HIDDEN_SIZES
}
DET_ROUTES = {
    arch: {
        key: route
        for key, route in routes.items()
        if key
        in {
            det.route_key(hidden, kind) for hidden in HIDDEN_SIZES for kind in det.KINDS
        }
    }
    for arch, routes in det.ROUTES.items()
}
DET_RECORD_SCALARS = {
    name: scalars for name, scalars in det.RECORD_SCALARS.items() if name in DET_MODULES
}
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


def _suffix(hidden: int) -> str:
    """The library suffix of a hidden size: none at 4096, ``__h<H>`` at the other sizes (the deterministic
    convention)."""
    return "" if hidden == HIDDEN else f"__h{hidden}"


def _atomic_split(arch: str, hidden: int = HIDDEN) -> str:
    """The atomic fn split of ``arch`` at ``hidden``: the form of the deterministic backward split
    (``fn_split_program(arch, 1, hidden)``), launched as the token-owner pass's programmatic dependent (``_dep``)."""
    return (
        f"{ND_PREFIX}fn_split_atomic__{det.fn_split_program(arch, 1, hidden)}_dep"
        + _suffix(hidden)
    )


def _slab(hidden: int = HIDDEN) -> str:
    """The column slab's ``dynr`` record of ``hidden``."""
    return SLAB + _suffix(hidden)


def _lpt(hidden: int = HIDDEN) -> str:
    """The column slab's two-size ``lpt`` record of ``hidden``."""
    return LPT + _suffix(hidden)


def _slab_os(hidden: int = HIDDEN) -> str:
    """The column slab's slab-fastest ``dynr_os`` record of ``hidden``."""
    return SLAB_OS + _suffix(hidden)


def _lpt_os(hidden: int = HIDDEN) -> str:
    """The column slab's slab-fastest two-size ``lpt_os`` record of ``hidden``."""
    return LPT_OS + _suffix(hidden)


def _slabs(arch: str, hidden: int = HIDDEN) -> list[str]:
    """The column slab's records on the ``bwd_finish`` route of ``arch`` at ``hidden``, in route order: ``dynr``,
    ``lpt``, ``dynr_os``, ``lpt_os``, each where the architecture has the record of the size (a size's ``_os``
    record exists only where some architecture's ``BWD_SLAB_ORDER`` of the size has a ``"slab"`` band)."""
    return [
        name
        for name in (_slab(hidden), _lpt(hidden), _slab_os(hidden), _lpt_os(hidden))
        if arch in ATOMIC.get(name, ())
    ]


def _lpt_table(hidden: int = HIDDEN) -> dict:
    """The two-size rule's calibration per architecture at ``hidden``: ``BWD_SLAB_LPT`` at 4096,
    ``BWD_SLAB_LPT_HIDDEN[hidden]`` at 5120 and 7168 (the dictionaries ``lpt_chunks`` reads at call time)."""
    return nd.BWD_SLAB_LPT if hidden == HIDDEN else nd.BWD_SLAB_LPT_HIDDEN[hidden]


def _band_at(T: int, bands: tuple):
    """The value of the first band of ``((t_upper, value), ..., (None, value))`` with ``T <= t_upper`` (``None``:
    every ``T``)."""
    return next(value for t_upper, value in bands if t_upper is None or t_upper >= T)


def _calibration_at(entry, T: int):
    """The ``(o_tiles, keep)`` (or ``None``) a calibration entry gives ``T``: the entry itself when it is ``None`` or
    one pair, else its token-count band of ``T``."""
    if entry is None or not isinstance(entry[0], tuple):
        return entry
    return _band_at(T, entry)


def _program_at(T: int, sm: int, arch: str, hidden: int = HIDDEN) -> str:
    """The column slab's program at ``T`` under the tables read at call time: ``lpt`` where ``lpt_chunks`` gives a
    geometry, else ``dynr``; ``_os`` appended where the size's ``BWD_SLAB_ORDER`` band of ``T`` is ``"slab"``."""
    base = "lpt" if nd.lpt_chunks(T, sm, arch, hidden=hidden) else "dynr"
    order = _band_at(T, nd.BWD_SLAB_ORDER[hidden][arch])
    return base + ("_os" if order == "slab" else "")


def _record_of(program: str, hidden: int = HIDDEN) -> str:
    """The column-slab record of ``hidden`` whose program key is ``program``."""
    names = {"dynr": SLAB, "lpt": LPT, "dynr_os": SLAB_OS, "lpt_os": LPT_OS}
    return names[program] + _suffix(hidden)


def _slab_record(T: int, sm: int, arch: str, hidden: int = HIDDEN) -> str:
    """The column-slab record the ``bwd_finish`` selection holds at ``T`` (:func:`_program_at`)."""
    return _record_of(_program_at(T, sm, arch, hidden), hidden)


def _route_hidden(key: str) -> int:
    """The hidden size of a route key ``<kind>_h<hidden>``."""
    return int(key.rsplit("_h", 1)[1])


def _arch_sizes(arch: str) -> tuple[int, ...]:
    """The hidden sizes of ``arch``: the deterministic package's (this package routes every size it does)."""
    return tuple(det.supported_hidden_sizes(arch))


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
    """``MODULES`` = every deterministic record of every hidden size but ``bwd_slab__*`` / ``bwd_reduce__*`` and the
    backward fn splits under the ``cake_mhc_train_`` prefix (the record equal field for field after the rename:
    arches, hidden size, build knobs, launch, arguments, rules) plus the programs of this package (``ATOMIC``); the
    dropped records are exactly the deterministic column-slab and reduction roles, and the backward fn splits are the
    only other records without a copy."""
    assert tuple(det.HIDDEN_SIZES) == HIDDEN_SIZES
    assert {record["hidden"] for record in DET_MODULES.values()} == set(HIDDEN_SIZES)
    assert DET_MODULES == det.MODULES
    dropped = {name for name in DET_MODULES if _dropped(name)}
    assert dropped and {_det_role(name) for name in dropped} == set(DROPPED_ROLES)
    assert not [
        name
        for name in DET_MODULES
        if name not in dropped and _det_role(name) in DROPPED_ROLES
    ]
    backward_splits = {
        name for name in DET_MODULES if _det_role(name) == "fn_split_bwd"
    }
    assert backward_splits and not backward_splits & dropped
    copied = {
        _renamed(name): name
        for name in DET_MODULES
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
        assert record["library"].endswith(_suffix(record["hidden"])), name


@pytest.mark.cpu_only
def test_atomic_records_keep_the_deterministic_launch() -> None:
    """The programs of this package, at every hidden size: named after their library and kernel symbol (``__h<H>``
    at 5120 and 7168); the fn splits keep the launch and fixed grid of the deterministic transposed-layout split of
    their form and size but for the static PDL attribute (``use_pdl`` True: they follow the token-owner pass and wait
    for it) and bind ``fn``, ``fnTp``, ``dfn`` and ``accumulate`` (no ``fnp``, no ``want_t``); the column slab keeps
    the launch and grid rule of the deterministic ``dynr`` record of its size (``slabs`` = H / 64), binds its
    arguments without ``dfn_part`` and adds ``dfn``, ``param_partial``, ``dscale``, ``dbase``, ``chunks`` and
    ``accumulate``.  At every hidden size the column slab's records select by ``bwd_slab_atomic_program`` (programs
    ``dynr`` / ``lpt`` / ``dynr_os`` / ``lpt_os``: the last ``__`` part of the name without the size suffix; on
    sm_107a, whose route holds the ``dynr`` and ``lpt`` records, and on sm_103a at 5120 and 7168, whose route holds
    the ``dynr`` and ``dynr_os`` records, the rule names one of these at every token count); the ``lpt`` record of a
    size is its ``dynr`` record with the ``lpt_chunks`` chunk rule, its chunk scalars ``P`` / ``rpc`` bound to the
    ``lpt_*`` host values and ``p_big`` / ``rpc_small`` after the ticket, on the architectures the size's table
    calibrates (``BWD_SLAB_LPT`` at 4096, ``BWD_SLAB_LPT_HIDDEN`` at 5120 and 7168: a calibration without the record
    would select no column slab, a record without one would never run); the ``dynr_os`` / ``lpt_os`` record of a size,
    where the size has one, is its ``dynr`` / ``lpt`` record (the slab-fastest build has the same kernel parameters,
    launch and chunk rule) under its own name and program key, on the architectures of that record whose
    ``BWD_SLAB_ORDER`` of the size has a ``"slab"`` band."""
    for name, arches in ATOMIC.items():
        record = nd.MODULES[name]
        hidden = record["hidden"]
        assert tuple(record["arches"]) == arches, name
        assert record["kernel"] == record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert hidden in HIDDEN_SIZES and name.endswith(_suffix(hidden)), name
        assert record["ffi_entry"] == "run", name
        if record["role"] == "bwd_slab":
            assert record["select_rule"] == {
                "kind": "program",
                "program": name.removesuffix(_suffix(hidden)).rsplit("__", 1)[1],
                "rule": "bwd_slab_atomic_program",
            }, name
        else:
            assert record["select_rule"] is None, name
        assert record["workspace_rule"] == {"kind": "regions", "workspace": "bwd"}
        assert name not in nd.RECORD_SCALARS, name
    for arch in det.ARCHES:
        for hidden in _arch_sizes(arch):
            what = (arch, hidden)
            split = nd.MODULES[_atomic_split(arch, hidden)]
            assert arch in split["arches"] and split["role"] == "fn_split_bwd", what
            assert split["hidden"] == hidden, what
            det_splits = [
                n
                for n in det.route_modules(arch, hidden, "bwd_finish")
                if _det_role(n) == "fn_split_bwd"
            ]
            assert len(det_splits) == 1, what
            reference = det.MODULES[det_splits[0]]
            assert reference["launch"]["use_pdl"] is False, what
            assert split["launch"] == {**reference["launch"], "use_pdl": True}, what
            assert split["grid_rule"] == reference["grid_rule"], what
            assert split["grid_rule"] == {
                "kind": "fixed",
                "ctas": det.fn_split_grid(arch, 1, hidden),
            }, what
            assert set(split["logical"].values()) == {"fn", "fnTp", "dfn", "accumulate"}
            assert [kind for kind, _ in split["arg_plan"]][-3:] == ["grid"] * 3
    added = {"dfn", "param_partial", "dscale", "dbase", "chunks", "accumulate"}
    for hidden in HIDDEN_SIZES:
        slab, dynr = nd.MODULES[_slab(hidden)], det.MODULES[DET_DYNR + _suffix(hidden)]
        assert slab["role"] == "bwd_slab" and slab["hidden"] == hidden, hidden
        assert slab["launch"] == dynr["launch"], hidden
        assert slab["launch"]["use_pdl"] == "runtime", hidden
        assert slab["launch"]["block"][0] == 384, hidden
        assert slab["grid_rule"] == dynr["grid_rule"], hidden
        assert slab["grid_rule"] == {
            "kind": "slab_chunks",
            "slabs": hidden // det.SLAB_D,
            "chunks_rule": "k7_chunks",
        }, hidden
        assert (
            set(slab["logical"].values())
            == (set(dynr["logical"].values()) - {"dfn_part"}) | added
        ), hidden
        kept = [entry for entry in dynr["arg_plan"] if entry != ["buffer", "dfn_part"]]
        assert [entry for entry in slab["arg_plan"] if entry[1] not in added] == kept
    for hidden in HIDDEN_SIZES:
        slab, lpt = nd.MODULES[_slab(hidden)], nd.MODULES[_lpt(hidden)]
        assert lpt["role"] == "bwd_slab" and lpt["hidden"] == hidden, hidden
        assert set(lpt["arches"]) <= set(slab["arches"]), hidden
        table = _lpt_table(hidden)
        assert set(table) == set(det.ARCHES), hidden
        assert set(lpt["arches"]) == {
            arch for arch, calibration in table.items() if calibration is not None
        }, hidden
        assert lpt["launch"] == slab["launch"], hidden
        assert lpt["grid_rule"] == {**slab["grid_rule"], "chunks_rule": "lpt_chunks"}
        assert lpt["arg_plan"][:-4] == slab["arg_plan"][:-4] + [
            ["parameter", "p_big"],
            ["parameter", "rpc_small"],
        ], hidden
        assert slab["arg_plan"][-5] == ["buffer", "ticket"], hidden
        assert lpt["arg_plan"][-4:] == slab["arg_plan"][-4:], hidden
        assert lpt["logical"] == {
            **slab["logical"],
            "P": "lpt_parts",
            "rpc": "lpt_rows_per_chunk",
            "p_big": "lpt_big_parts",
            "rpc_small": "lpt_small_rows_per_chunk",
        }, hidden
    for hidden in HIDDEN_SIZES:
        for base, name in (
            (_slab(hidden), _slab_os(hidden)),
            (_lpt(hidden), _lpt_os(hidden)),
        ):
            if name not in ATOMIC:
                assert name not in nd.MODULES, name
                continue
            record, reference = nd.MODULES[name], nd.MODULES[base]
            assert set(record["arches"]) <= set(reference["arches"]), name
            program = reference["select_rule"]["program"] + "_os"
            assert record == {
                **reference,
                "arches": record["arches"],
                "kernel": name,
                "kernel_symbol": f"kernel_{name}",
                "library": name,
                "select_rule": {**reference["select_rule"], "program": program},
            }, name


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
    without ``dfn_part``; ``LOGICAL_SCALARS`` = the deterministic one and the lpt record's chunk scalars; tensors as
    buffers, scalars as parameters), one logical name per kernel parameter, the
    per-call PDL argument exactly when ``use_pdl`` is ``"runtime"`` (just before the grid).  The host binds every
    argument of every routed record: each logical name is a call value, a workspace region, a host-plan scalar of
    the route's fast plan or a scalar the record fixes (the run-time-count records also take ``sinkhorn_iters``,
    which the full path passes), and every workspace value names a region of the route's workspace."""
    assert list(nd.LOGICAL_TENSORS) == [
        t for t in det.LOGICAL_TENSORS if t != "dfn_part"
    ]
    assert "dfn_part" in det.LOGICAL_TENSORS
    assert nd.LOGICAL_SCALARS == det.LOGICAL_SCALARS + LPT_HOST_KEYS
    assert set(pkg._HOST_VALUES["bwd_finish"]) >= set(LPT_HOST_KEYS)
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
            assert nd.route_key(_route_hidden(key), kind) == key
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
    """``ROUTES``: the deterministic architectures and route keys of every hidden size (5120 and 7168 on sm_100a /
    sm_103a only); every route the renamed deterministic route except ``bwd_finish``, which is the deterministic
    token-owner records (renamed, in route order), the atomic fn split of the architecture and size and the column
    slab's forms of the size (``dynr``, ``lpt``, ``dynr_os``, ``lpt_os``; on sm_107a ``dynr`` and ``lpt``; on sm_103a
    at 5120 and 7168 ``dynr`` and ``dynr_os``).  Every record is routed, on exactly its architectures."""
    assert set(nd.ROUTES) == set(det.ROUTES) == set(det.ARCHES)
    routed: dict[str, set] = {}
    for arch in det.ARCHES:
        sizes = _arch_sizes(arch)
        assert sizes == (HIDDEN_SIZES if arch in NEW_HIDDEN_ARCHES else (HIDDEN,)), arch
        assert set(DET_ROUTES[arch]) == {
            det.route_key(hidden, kind) for hidden in sizes for kind in det.KINDS
        }
        for key, det_route in DET_ROUTES[arch].items():
            assert {det.MODULES[n]["hidden"] for n in det_route} == {
                _route_hidden(key)
            }, (arch, key)
        assert set(nd.ROUTES[arch]) == set(DET_ROUTES[arch]), arch
        for key, det_route in DET_ROUTES[arch].items():
            route = nd.ROUTES[arch][key]
            for name in route:
                routed.setdefault(name, set()).add(arch)
            if not key.startswith("bwd_finish_h"):
                assert route == _renamed(det_route), (arch, key)
                continue
            hidden = _route_hidden(key)
            tokens = [_renamed(n) for n in det_route if _det_role(n) == "bwd_token"]
            assert tokens, (arch, key)
            slabs = _slabs(arch, hidden)
            assert route == [*tokens, _atomic_split(arch, hidden), *slabs], (arch, key)
            assert {nd.MODULES[n]["hidden"] for n in route} == {hidden}, (arch, key)
            assert {_det_role(n) for n in det_route} == {
                "fn_split_bwd",
                "bwd_token",
                *DROPPED_ROLES,
            }, (arch, key)
    for name, arches in routed.items():
        assert arches == set(nd.MODULES[name]["arches"]), name
    assert set(nd.MODULES) == set(routed), sorted(set(nd.MODULES) ^ set(routed))


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_selection_is_the_deterministic_selection_but_bwd_finish(
    arch: str, sm: int
) -> None:
    """At every hidden size of the board's architecture, on every token count of record at the count of record and at
    run-time counts: every kind selects the renamed deterministic records, except ``bwd_finish``: (the deterministic
    token-owner record(s) of the call renamed, the atomic fn split of the size, the column slab's form of the call --
    the size's ``lpt`` record where ``lpt_chunks`` gives a geometry of the size under the table of record, else its
    ``dynr`` record, as its ``_os`` copy where the size's ``BWD_SLAB_ORDER`` band is ``"slab"``) -- one fn split, one
    column slab, no reduction."""
    for hidden, T in ((h, T) for h in _arch_sizes(arch) for T in T_VALUES):
        for iters in (det.SINKHORN_ITERS, *RUN_TIME_ITERS):
            for kind in det.KINDS:
                what = (arch, sm, hidden, T, iters, kind)
                det_names = det.selected_modules(
                    arch, hidden, kind, rows=T, sm_count=sm, sinkhorn_iters=iters
                )
                names = nd.selected_modules(
                    arch, hidden, kind, rows=T, sm_count=sm, sinkhorn_iters=iters
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
                slab = _slab_record(T, sm, arch, hidden)
                assert names == (*tokens, _atomic_split(arch, hidden), slab), (
                    what,
                    names,
                )
                run_time = (nd.MODULES[tokens[0]]["select_rule"] or {}).get("kind")
                assert (run_time == "run_time_iters") == (
                    iters != det.SINKHORN_ITERS
                ), what


def _possible_programs(arch: str, hidden: int) -> set[str]:
    """The programs ``bwd_slab_atomic_program`` can name on ``arch`` at ``hidden`` under the tables read at call
    time: on each stretch of token counts where both the size's ``BWD_SLAB_ORDER`` band and its calibration band are
    constant, ``dynr`` (the screen may pick no geometry) and ``lpt`` where the calibration is not ``None``, each with
    ``_os`` where the order is ``"slab"``."""
    order = nd.BWD_SLAB_ORDER[hidden][arch]
    entry = _lpt_table(hidden)[arch]
    bands = entry if entry is not None and isinstance(entry[0], tuple) else ()
    edges = sorted({t for t, _ in (*order, *bands) if t is not None})
    programs = set()
    for T in (1, *(t + 1 for t in edges)):
        suffix = "_os" if _band_at(T, order) == "slab" else ""
        programs.add("dynr" + suffix)
        if _calibration_at(entry, T) is not None:
            programs.add("lpt" + suffix)
    return programs


@pytest.mark.cpu_only
def test_slab_tables_route_every_program_they_name() -> None:
    """On every architecture and routed hidden size: every program the column slab's tables can name (``BWD_SLAB_LPT`` /
    ``BWD_SLAB_LPT_HIDDEN`` with ``BWD_SLAB_ORDER``) has its record on the ``bwd_finish`` route (a program without one
    would select no column slab), one record per program; and every ``_os`` record on the route is one the tables can
    name: ``dynr_os`` needs a ``"slab"`` band of the size, ``lpt_os`` a calibration band other than ``None`` that
    overlaps one.  A size an architecture does not route has ``"chunk"`` bands alone (a ``"slab"`` band would name a
    slab-fastest program that has no record there)."""
    for hidden, by_arch in nd.BWD_SLAB_ORDER.items():
        for arch, bands in by_arch.items():
            if hidden not in _arch_sizes(arch):
                assert {order for _bound, order in bands} == {"chunk"}, (
                    arch,
                    hidden,
                    bands,
                )
    checked = 0
    for arch in det.ARCHES:
        for hidden in _arch_sizes(arch):
            what = (arch, hidden)
            slabs = [
                name
                for name in nd.route_modules(arch, hidden, "bwd_finish")
                if nd.MODULES[name]["role"] == "bwd_slab"
            ]
            routed = {
                nd.MODULES[name]["select_rule"]["program"]: name for name in slabs
            }
            assert len(routed) == len(slabs), (what, slabs)
            possible = _possible_programs(arch, hidden)
            assert possible <= set(routed), (what, sorted(possible - set(routed)))
            for program, name in routed.items():
                assert name == _record_of(program, hidden), (what, name)
                if program.endswith("_os"):
                    assert program in possible, (what, name, sorted(possible))
                    checked += 1
    assert checked


@pytest.mark.cpu_only
def test_selection_refuses_a_column_slab_the_route_lacks(monkeypatch) -> None:
    """A call whose column-slab program has no record on its route (here ``lpt_os`` on sm_103a at hidden 5120, whose
    route holds ``dynr`` and ``dynr_os``) is a ``ValueError`` of the ``bwd_finish`` selection, never a backward
    without the column slab; the other kinds select as before."""
    arch, sm, hidden, T = "sm_103a", 152, 5120, 2049
    before = {
        kind: nd.selected_modules(arch, hidden, kind, rows=T, sm_count=sm)
        for kind in det.KINDS
    }
    assert _lpt_os(hidden) not in nd.route_modules(arch, hidden, "bwd_finish")
    monkeypatch.setattr(nd, "bwd_slab_atomic_program", lambda *_args: "lpt_os")
    with pytest.raises(ValueError, match="selects 0 column-slab records"):
        nd.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
    for kind in det.KINDS:
        if kind != "bwd_finish":
            assert (
                nd.selected_modules(arch, hidden, kind, rows=T, sm_count=sm)
                == before[kind]
            ), kind


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_pdl_follows_the_deterministic_edges(arch: str, sm: int) -> None:
    """``PDL_EDGES`` = the deterministic edges without ``K7->K8``, the column slab's stem
    ``cake_mhc_train_bwd_slab_atomic`` on ``K6->K7`` (every form and size); ``PDL_OFF_BANDS`` = the deterministic
    bands without ``K7->K8`` (one table for every hidden size, as in the deterministic package).  On every path and
    token count (the counts of record and every band boundary) the ``enable_pdl`` of the column-slab records of each
    hidden size equals every deterministic column-slab record's of the route of that size, the fn splits take no
    per-call argument (their attribute is static), and every copied record reads the deterministic value."""
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
    sizes = _arch_sizes(arch)
    # per hidden size: (this package's column-slab records, its atomic fn split, the deterministic column slabs)
    slabs = {
        hidden: (
            _slabs(arch, hidden),
            _atomic_split(arch, hidden),
            [
                n
                for n in det.route_modules(arch, hidden, "bwd_finish")
                if _det_role(n) == "bwd_slab"
            ],
        )
        for hidden in sizes
    }
    for hidden, (names, _split, det_slabs) in slabs.items():
        assert det_slabs, (arch, hidden)
        for name in names:
            assert nd.pdl_edge(nd.MODULES[name]["library"]) == "K6->K7", name
    boundaries = set()
    for bands in det.PDL_OFF_BANDS.get(arch, {}).get("K6->K7", {}).values():
        for above, at_most in bands:
            boundaries |= {above, above + 1}
            if at_most is not None:
                boundaries |= {at_most, at_most + 1}
    copied = [
        n
        for route in DET_ROUTES[arch].values()
        for n in route
        if not _dropped(n) and _det_role(n) != "fn_split_bwd"
    ]
    for T in sorted(set(T_VALUES) | {t for t in boundaries if t > 0}):
        for path in det.PDL_PATHS:
            value = nd.enable_pdl(SLAB, T, arch, path)
            assert value in (0, 1)
            for hidden, (names, split, det_slabs) in slabs.items():
                for name in names:
                    assert nd.enable_pdl(name, T, arch, path) == value, (T, path, name)
                for name in det_slabs:
                    assert value == det.enable_pdl(name, T, arch, path), (T, path, name)
                assert nd.enable_pdl(split, T, arch, path) == 1, (T, path, hidden)
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
    """At every hidden size of the board's architecture: ``fwd`` is the deterministic forward workspace; the ``bwd``
    regions are the deterministic ``bwd`` regions but the last (``dfn_part``, ``P x 24 x 4H`` FP32) at identical
    offsets, sizes, dtypes and shapes, the ticket region first; the total is the deterministic total less the rounded
    ``dfn_part`` bytes; ``T == 0`` has no regions."""
    assert set(nd.WORKSPACE_REGIONS) == set(det.WORKSPACE_REGIONS) == {"fwd", "bwd"}
    assert nd.WORKSPACE_REGIONS["fwd"] == det.WORKSPACE_REGIONS["fwd"]
    assert det.WORKSPACE_REGIONS["bwd"][-1]["name"] == "dfn_part"
    assert nd.WORKSPACE_REGIONS["bwd"] == det.WORKSPACE_REGIONS["bwd"][:-1]
    assert nd.WORKSPACE_OF_KIND == det.WORKSPACE_OF_KIND
    for hidden, T in ((h, T) for h in _arch_sizes(arch) for T in det.RULES_T_VALUES):
        what = (hidden, T)
        assert nd.workspace_regions(
            "fwd", T, sm, arch, hidden
        ) == det.workspace_regions("fwd", T, sm, arch, hidden), what
        layout = nd.workspace_regions("bwd", T, sm, arch, hidden)
        reference = det.workspace_regions("bwd", T, sm, arch, hidden)
        assert layout["total_bytes"] == nd.workspace_bytes(
            "bwd", T, sm, arch, hidden
        ), what
        if T == 0:
            assert layout == {"regions": {}, "total_bytes": 0}
            continue
        names = list(reference["regions"])
        assert names[-1] == "dfn_part" and list(layout["regions"]) == names[:-1], what
        for name, region in layout["regions"].items():
            assert region == reference["regions"][name], (what, name)
        assert layout["regions"]["ticket"]["offset"] == 0
        assert layout["regions"]["ticket"]["bytes"] == 4 * det.TICKET_WORDS
        part = reference["regions"]["dfn_part"]
        parts = det.k7_chunks(T, sm, hidden=hidden)[0]
        assert part["shape"] == (parts, 24, 4 * hidden), what
        assert part["bytes"] == parts * 24 * 4 * hidden * 4, what
        assert layout["total_bytes"] == reference["total_bytes"] - _round256(
            part["bytes"]
        ), what
        assert layout["total_bytes"] % 256 == 0, what
    with pytest.raises(ValueError):
        nd.workspace_regions("bwd", -1, sm, arch)
    with pytest.raises(ValueError):
        nd.workspace_regions("dfn", 1, sm, arch)
    with pytest.raises(ValueError):
        nd.workspace_regions("bwd", 1, sm, arch, 6144)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_workspace_envelope_covers_every_smaller_token_count(
    arch: str, sm: int
) -> None:
    """At every hidden size of the board's architecture: ``workspace_bytes_upto("bwd", T)`` is non-decreasing, at
    least the exact size of every token count up to ``T`` and at most the deterministic envelope (a buffer of the
    deterministic sizing serves this package); 0 at ``T == 0``; the forward envelope is the deterministic one.  Every
    token count below 4200 at hidden 4096, every seventh at 5120 and 7168."""
    for hidden in _arch_sizes(arch):
        previous = 0
        for T in range(0, 4200, 1 if hidden == HIDDEN else 7):
            what = (hidden, T)
            envelope = nd.workspace_bytes_upto("bwd", T, sm, arch, hidden)
            assert envelope >= previous, what
            assert envelope >= nd.workspace_bytes("bwd", T, sm, arch, hidden), what
            assert envelope <= det.workspace_bytes_upto("bwd", T, sm, arch, hidden), (
                what
            )
            previous = envelope
        assert nd.workspace_bytes_upto("bwd", 0, sm, arch, hidden) == 0
        for rows in (64, 65, 1000, 2049, 8192, 16263, 20001, 65536):
            what = (hidden, rows)
            envelope = nd.workspace_bytes_upto("bwd", rows, sm, arch, hidden)
            assert envelope <= det.workspace_bytes_upto(
                "bwd", rows, sm, arch, hidden
            ), what
            assert nd.workspace_bytes_upto(
                "fwd", rows, sm, arch, hidden
            ) == det.workspace_bytes_upto("fwd", rows, sm, arch, hidden), what
            for T in sorted(
                {t for t in det.RULES_T_VALUES if t <= rows}
                | {rows, rows - 1, (rows + 1) // 2}
            ):
                assert envelope >= nd.workspace_bytes("bwd", T, sm, arch, hidden), (
                    what,
                    T,
                )


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
    "K5_FORM_BANDS", "K5_FORM_ROWS", "FWD_FINISH_PROGRAM", "FWD_FINISH_ROWS", "FWD_FINISH_COL_SPLIT", "K3_TILING",
    "K6_CTAS_PER_SM", "PDL_PATHS", "SELECT_RULE_KINDS", "RUN_TIME_ITERS_ROLES",
)  # fmt: skip
# The deterministic column-slab and reduction rules, absent here.
_DROPPED_RULES = (
    "BWD_SLAB_PROGRAM", "BWD_SLAB_BANDS", "BWD_SLAB_FIT", "BWD_REDUCE_PROGRAM", "BWD_REDUCE_REGIMES",
    "BWD_REDUCE_PARTS", "BWD_REDUCE_GRID", "bwd_slab_program", "bwd_reduce_program", "bwd_reduce_grid",
)  # fmt: skip
# This package's own column-slab rule, absent from the deterministic registry.
_OWN_RULES = (
    "BWD_SLAB_LPT",
    "BWD_SLAB_LPT_HIDDEN",
    "BWD_SLAB_ORDER",
    "lpt_chunks",
    "bwd_slab_order",
    "bwd_slab_atomic_program",
    "slab_plan",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_host_rules_are_copies_of_the_deterministic_ones(arch: str, sm: int) -> None:
    """The constants and tables of the host rules equal the deterministic ones; the rule functions are this
    package's own objects with the deterministic outputs on every token count of record; the deterministic
    column-slab and reduction rules are absent and ``PROGRAM_RULES`` drops them for this package's column-slab rule
    (``bwd_slab_atomic_program``, with ``BWD_SLAB_LPT``, ``BWD_SLAB_LPT_HIDDEN``, ``BWD_SLAB_ORDER``,
    ``lpt_chunks``, ``bwd_slab_order`` and ``slab_plan``: none of them in the deterministic registry); a rules row is
    the deterministic row without its column-slab and reduction program columns, with this package's backward
    workspace bytes."""
    for name in _SHARED_TABLES:
        assert getattr(nd, name) == getattr(det, name), name
    # The hidden sizes: the deterministic ones; the per-size table names are the deterministic ones without the
    # column-slab and reduction tables, and the tables of each size other than 4096 the deterministic tables of those
    # names (the tables of record above are the H 4096 ones).
    assert tuple(nd.HIDDEN_SIZES) == tuple(det.HIDDEN_SIZES) == HIDDEN_SIZES
    kept = tuple(name for name in det.H_TABLE_NAMES if name not in _DROPPED_RULES)
    assert kept == nd.H_TABLE_NAMES
    assert HIDDEN not in det.H_TABLES and set(det.H_TABLES) == set(NEW_HIDDEN_SIZES)
    assert {
        hidden: {name: det.H_TABLES[hidden][name] for name in nd.H_TABLE_NAMES}
        for hidden in NEW_HIDDEN_SIZES
    } == nd.H_TABLES
    for name in _DROPPED_RULES:
        assert hasattr(det, name) and not hasattr(nd, name), name
    for name in _OWN_RULES:
        assert hasattr(nd, name) and not hasattr(det, name), name
    own_rules = (
        *(rule for rule in det.PROGRAM_RULES if rule not in _DROPPED_RULES),
        "bwd_slab_atomic_program",
    )
    assert own_rules == nd.PROGRAM_RULES
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
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_host_rules_at_the_new_hidden_sizes(arch: str, sm: int) -> None:
    """At hidden 5120 and 7168 on every board (the rules are functions of the size's tables; sm_107a routes neither
    size): a rules row is the deterministic row without its column-slab and reduction program columns, with this
    package's backward workspace bytes, on every token count of record; the host plan and every copied rule give the
    deterministic outputs; the per-size tables are read through ``_tab`` (a size it does not spell is a ``KeyError``,
    never the H 4096 value; the column slab's ``BWD_SLAB_LPT`` / ``BWD_SLAB_LPT_HIDDEN`` / ``BWD_SLAB_ORDER`` are no
    per-size tables of ``_tab``).  ``rules_table()`` holds the H 4096 rows, then each other size's."""
    dropped_columns = {"bwd_slab_program", "bwd_reduce_program", "bwd_reduce_grid"}
    for hidden in NEW_HIDDEN_SIZES:
        for name in nd.H_TABLE_NAMES:
            assert nd._tab(name, hidden) is nd.H_TABLES[hidden][name], (hidden, name)
        for layout in (0, 1):
            assert nd.fn_split_program(arch, layout, hidden) == det.fn_split_program(
                arch, layout, hidden
            )
            assert nd.fn_split_grid(arch, layout, hidden) == det.fn_split_grid(
                arch, layout, hidden
            )
        assert nd.fwd_finish_program(arch, hidden) == det.fwd_finish_program(
            arch, hidden
        )
        for T in det.RULES_T_VALUES:
            what = (hidden, T)
            row = nd.rules_row(T, sm, arch, hidden)
            reference = det.rules_row(T, sm, arch, hidden)
            expected = {k: v for k, v in reference.items() if k not in dropped_columns}
            expected["bwd_workspace_bytes"] = nd.workspace_bytes(
                "bwd", T, sm, arch, hidden
            )
            assert row == expected, what
            if T == 0:
                continue
            assert nd.host_plan(T, sm, arch, hidden) == det.host_plan(
                T, sm, arch, hidden
            ), what
            for rule in (
                "fwd_map_program",
                "fwd_coef_program",
                "bwd_token_program",
                "k5_program",
                "k5_grid",
            ):
                assert getattr(nd, rule)(T, sm, arch, hidden) == getattr(det, rule)(
                    T, sm, arch, hidden
                ), (rule, what)
            assert nd.k7_chunks(T, sm, hidden=hidden) == det.k7_chunks(
                T, sm, hidden=hidden
            ), what
            assert nd.k6_grid(T, sm, hidden) == det.k6_grid(T, sm, hidden), what
            assert nd.k3_grid(T, hidden) == det.k3_grid(T, hidden), what
    for name in nd.H_TABLE_NAMES:
        assert nd._tab(name, HIDDEN) is getattr(nd, name), name
        with pytest.raises(KeyError):
            nd._tab(name, 6144)
    for name in ("BWD_SLAB_LPT", "BWD_SLAB_LPT_HIDDEN", "BWD_SLAB_ORDER"):
        for hidden in HIDDEN_SIZES:
            with pytest.raises(KeyError):
                nd._tab(name, hidden)
    # the token owner's resident CTAs per SM (G6 = min(K6_CTAS_PER_SM x SMs, ceil(T / 8))): two at 5120, one at 7168
    assert nd._tab("K6_CTAS_PER_SM", 5120) == 2 and nd._tab("K6_CTAS_PER_SM", 7168) == 1
    table = nd.rules_table()
    per_size = len(set(det.RULES_T_VALUES)) * len(det.RULES_BOARDS)
    assert [row["hidden"] for row in table] == [
        hidden for hidden in HIDDEN_SIZES for _ in range(per_size)
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arch", "sm"),
    [board for board in det.RULES_BOARDS if board[0] in NEW_HIDDEN_ARCHES],
)
def test_launch_grids_at_the_new_hidden_sizes(arch: str, sm: int, monkeypatch) -> None:
    """At hidden 5120 and 7168, on every token count of record, every record a call selects launches the grid of the
    deterministic record it copies or replaces: a copy the renamed deterministic record's grid, the atomic fn split
    the deterministic backward split's fixed grid (the rules row's ``fn_split_grid_t``: 320 / 448 CTAs), the column
    slab's ``dynr`` / ``dynr_os`` record ``min(SMs, H / 64 x P)`` CTAs, the deterministic ``dynr`` record's grid (the
    rules row's ``k7_grid``), and its ``lpt`` / ``lpt_os`` record, where the call selects it, ``min(SMs, H / 64 x
    lpt_parts)`` CTAs over the size's slabs (:func:`test_forced_geometry_is_read_at_call_time` checks that
    ``launch_grid`` reads the record's hidden size)."""
    for module in (nd, det):
        monkeypatch.setattr(module, "sm_count", lambda _index: sm)
        monkeypatch.setattr(module, "device_arch", lambda _index: arch)
    checked = 0
    for hidden in NEW_HIDDEN_SIZES:
        for T in T_VALUES:
            row = det.rules_row(T, sm, arch, hidden)
            for kind in det.KINDS:
                what = (hidden, T, kind)
                det_names = det.selected_modules(
                    arch, hidden, kind, rows=T, sm_count=sm
                )
                for name in nd.selected_modules(
                    arch, hidden, kind, rows=T, sm_count=sm
                ):
                    record = nd.MODULES[name]
                    grid = nd.launch_grid(record, T, device_index=0)
                    if record["role"] == "bwd_slab" and name in (
                        _lpt(hidden),
                        _lpt_os(hidden),
                    ):
                        assert nd.lpt_chunks(T, sm, arch, hidden=hidden), (what, name)
                        parts = nd.slab_plan(T, sm, arch, hidden)["lpt_parts"]
                        want = (min(sm, hidden // det.SLAB_D * parts), 1, 1)
                    elif record["role"] == "bwd_slab":
                        assert grid == (
                            min(sm, hidden // det.SLAB_D * row["P"]),
                            1,
                            1,
                        ), (what, name)
                        want = (row["k7_grid"], 1, 1)
                    elif record["role"] == "fn_split_bwd":
                        want = (row["fn_split_grid_t"], 1, 1)
                    else:
                        det_name = name.replace(ND_PREFIX, DET_PREFIX)
                        assert det_name in det_names, (what, name)
                        want = det.launch_grid(det.MODULES[det_name], T, device_index=0)
                    assert grid == want, (what, name)
                    checked += 1
    assert checked


# ----------------------------------------------------------------------------------------------------------------
# The column slab's two-size rule and unit order (this package's own: BWD_SLAB_LPT, BWD_SLAB_LPT_HIDDEN,
# BWD_SLAB_ORDER, lpt_chunks, bwd_slab_order, bwd_slab_atomic_program, slab_plan)


def _clear_plans() -> None:
    """Drop the package's cached route plans (an experiment that edits ``BWD_SLAB_LPT`` / ``BWD_SLAB_LPT_HIDDEN`` /
    ``BWD_SLAB_ORDER`` or replaces ``lpt_chunks`` clears them on entry and exit)."""
    pkg._PLANS.clear()
    pkg._FAST_ROUTES.clear()


def _units(T: int, geometry: tuple, hidden: int = HIDDEN) -> list[int]:
    """The tile count of each unit of ``geometry`` at ``hidden`` (``S = H / 64`` slabs) in the kernel's draw order:
    unit ``u < S p_big`` is big chunk ``u % p_big``, unit ``S p_big + v`` small chunk ``v % (P - p_big)``; each chunk
    clipped to the T tokens."""
    P, rpc, p_big, rpc_small = geometry
    tiles, rb, rs, small = -(-T // 64), rpc // 64, rpc_small // 64, P - p_big
    slabs = hidden // det.SLAB_D
    big_tiles = [
        max(0, min(rb, tiles - (u % p_big) * rb)) for u in range(slabs * p_big)
    ]
    small_tiles = [
        max(0, min(rs, tiles - p_big * rb - (v % small) * rs))
        for v in range(slabs * small)
    ]
    return big_tiles + small_tiles


def _makespan(units: list[int], sm: int, o_tiles: float) -> float:
    """List scheduling without a heap: each unit, in order, to the first least loaded SM; the largest load."""
    loads = [0.0] * min(sm, len(units))
    for tiles in units:
        k = loads.index(min(loads))
        loads[k] = loads[k] + o_tiles + tiles
    return max(loads)


# (arch, sm, hidden, calibration): every hidden size of the board's architecture under its table of record, and under
# the kernel test's screen where the architecture has the size's lpt record (elsewhere a screen that fires would
# select no column slab).
_TWO_SIZE_CASES = [
    pytest.param(
        arch, sm, hidden, calibration, id=f"{arch}-{sm}-H{hidden}-{calibration}"
    )
    for arch, sm in det.RULES_BOARDS
    for hidden in det.supported_hidden_sizes(arch)
    for calibration in (
        ("record", "screen") if arch in ATOMIC[_lpt(hidden)] else ("record",)
    )
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "hidden", "calibration"), _TWO_SIZE_CASES)
def test_two_size_rule(arch: str, sm: int, hidden: int, calibration: str) -> None:
    """At every token count of record and of ``LPT_SWEEP``, at every hidden size of the board's architecture, under
    the size's table of record (``BWD_SLAB_LPT`` at 4096, ``BWD_SLAB_LPT_HIDDEN`` at 5120 and 7168) and under the
    kernel test's screen ``(1.0, 0.99)`` where the architecture has the size's ``lpt`` record: ``None`` never fires;
    ``bwd_slab_atomic_program`` is ``lpt`` exactly where ``lpt_chunks`` gives a geometry, else ``dynr``, with ``_os``
    where the size's ``BWD_SLAB_ORDER`` band is ``"slab"``, and the selection holds that one column-slab record of the
    size; ``slab_plan`` is the geometry, else the one-size chunks of ``host_plan``.  A geometry is 64-token
    multiples, ``1 <= p_big <= P <= 32``, big chunks within six tiles under the one-size chunks, covering the tokens:
    two sizes (the big chunks leave tokens, a smaller small chunk) or one (``p_big = P``, the last chunk holding
    tokens); its list-scheduling makespan over the size's ``H / 64`` slabs (a scheduler of this test) is at most
    ``keep`` times the one-size chunks'.  The firing token counts are printed; under the screen the rule fires on B200
    (148 SMs) at every size."""
    table = _lpt_table(hidden)
    value = table[arch] if calibration == "record" else LPT_SCREEN
    fired, counted = [], 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(table, arch, value)
        _clear_plans()
        try:
            for T in sorted(set(T_VALUES) | set(LPT_SWEEP)):
                counted += 1
                what = (arch, sm, hidden, calibration, T)
                geometry = nd.lpt_chunks(T, sm, arch, hidden=hidden)
                if _calibration_at(value, T) is None:
                    assert geometry is None, what
                order = _band_at(T, nd.BWD_SLAB_ORDER[hidden][arch])
                program = nd.bwd_slab_atomic_program(T, sm, arch, hidden)
                assert program == ("lpt" if geometry else "dynr") + (
                    "_os" if order == "slab" else ""
                ), what
                names = nd.selected_modules(
                    arch, hidden, "bwd_finish", rows=T, sm_count=sm
                )
                slabs = [n for n in names if nd.MODULES[n]["role"] == "bwd_slab"]
                assert slabs == [_record_of(program, hidden)], (what, names)
                plan = nd.host_plan(T, sm, arch, hidden)
                one = (
                    plan["parts"],
                    plan["rows_per_chunk"],
                    plan["parts"],
                    plan["rows_per_chunk"],
                )
                values = nd.slab_plan(T, sm, arch, hidden)
                assert tuple(values[k] for k in LPT_HOST_KEYS) == (geometry or one), (
                    what
                )
                if geometry is None:
                    continue
                fired.append((T, geometry))
                P, rpc, p_big, rpc_small = geometry
                rt = plan["rows_per_chunk"] // 64
                assert rpc % 64 == 0 and rpc_small % 64 == 0, what
                assert 1 <= p_big <= P <= 32, what
                assert max(1, rt - 6) <= rpc // 64 <= rt, what
                assert p_big * rpc + (P - p_big) * rpc_small >= T, what
                if p_big < P:
                    assert p_big * rpc < T and 0 < rpc_small < rpc, what
                else:
                    assert rpc_small == rpc and (P - 1) * rpc < T, what
                o_tiles, keep = _calibration_at(value, T)
                assert _makespan(
                    _units(T, geometry, hidden), sm, o_tiles
                ) <= keep * _makespan(_units(T, one, hidden), sm, o_tiles), what
        finally:
            _clear_plans()
    print(
        f"lpt_chunks {arch} / {sm} SMs H {hidden} at {calibration} {value}: fires at {len(fired)} of {counted} token "
        f"counts; at the counts of record {[(T, g) for T, g in fired if T in T_VALUES]}",
        flush=True,
    )
    if value is None:
        assert not fired
    if calibration == "screen" and (arch, sm) == ("sm_100a", 148):
        assert fired, (
            f"the screen fires at no token count on B200 at H {hidden}: the lpt record is never exercised"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "fires"), LPT_FIRES)
def test_two_size_rule_fires_where_calibrated(arch: str, sm: int, fires: dict) -> None:
    """Under the table of record the two-size schedule runs exactly at the benchmark token counts of ``LPT_FIRES``,
    with these geometries: a table that stops firing would ship dynr alone without a failing test.  The program is
    ``lpt`` or ``lpt_os`` there (``dynr`` / ``dynr_os`` elsewhere), ``_os`` exactly where the ``BWD_SLAB_ORDER`` band
    is ``"slab"``."""
    _clear_plans()
    try:
        got = {T: nd.lpt_chunks(T, sm, arch) for T in BENCH_T}
        programs = {T: nd.bwd_slab_atomic_program(T, sm, arch) for T in BENCH_T}
    finally:
        _clear_plans()
    assert {T: tuple(g) for T, g in got.items() if g is not None} == fires
    assert {T for T, p in programs.items() if p.removesuffix("_os") == "lpt"} == set(
        fires
    )
    for T, program in programs.items():
        slab_order = _band_at(T, nd.BWD_SLAB_ORDER[HIDDEN][arch]) == "slab"
        assert program.endswith("_os") == slab_order, (T, program)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "hidden", "fires"), LPT_FIRES_HIDDEN)
def test_two_size_rule_fires_where_calibrated_at_the_new_hidden_sizes(
    arch: str, sm: int, hidden: int, fires: dict
) -> None:
    """At hidden 5120 and 7168 under the table of record (``BWD_SLAB_LPT_HIDDEN``) the two-size schedule runs exactly
    at the token counts of ``LPT_FIRES_HIDDEN`` with these geometries, ``slab_plan`` the geometry there (on B200 the
    token counts the table's comment names; nowhere on the boards the size's table does not calibrate, which have no
    ``lpt`` record of the size).  The screen's plans are cached per hidden size: after the H 4096 plan of the same
    token count, SM count and ``o_tiles``, the size's plan is its own."""
    _clear_plans()
    try:
        got = {T: nd.lpt_chunks(T, sm, arch, hidden=hidden) for T in LPT_HIDDEN_T}
        programs = {
            T: nd.bwd_slab_atomic_program(T, sm, arch, hidden) for T in LPT_HIDDEN_T
        }
    finally:
        _clear_plans()
    assert {T: tuple(g) for T, g in got.items() if g is not None} == fires
    assert {T for T, p in programs.items() if p.removesuffix("_os") == "lpt"} == set(
        fires
    )
    for T, program in programs.items():
        slab_order = _band_at(T, nd.BWD_SLAB_ORDER[hidden][arch]) == "slab"
        assert program.endswith("_os") == slab_order, (T, program)
    assert bool(fires) == (arch in ATOMIC[_lpt(hidden)]), (arch, hidden)
    for T, geometry in fires.items():
        assert nd.slab_plan(T, sm, arch, hidden) == dict(
            zip(LPT_HOST_KEYS, geometry, strict=True)
        ), T
        o_tiles = float(_calibration_at(_lpt_table(hidden)[arch], T)[0])
        at_4096 = nd._lpt_plan(T, sm, o_tiles)
        plan = nd._lpt_plan(T, sm, o_tiles, hidden=hidden)
        assert plan[2] == geometry and plan != at_4096, (T, plan, at_4096)


def _heap_makespan(units: list[int], sm: int, o_tiles: float) -> float:
    """The kernel test's list scheduling, unit by unit on a heap of SM loads; the largest load."""
    loads = [(0.0, k) for k in range(min(sm, len(units)))]
    for tiles in units:
        load, k = heapq.heappop(loads)
        heapq.heappush(loads, (load + o_tiles + tiles, k))
    return max(load for load, _ in loads)


@pytest.mark.cpu_only
def test_two_size_makespan_runs_are_the_unit_by_unit_schedule() -> None:
    """``_lpt_makespan`` over the runs of ``_lpt_units`` (the SMs of one load level take a run's units together) is bit
    for bit the unit-by-unit list scheduling of every candidate geometry the screen ranks and of the one-size chunks,
    on 1 to 216 SMs and at integer and non-integer ``o_tiles``, over the ``H / 64`` slabs of every hidden size (fewer
    token and SM counts at 5120 and 7168); the runs hold every unit in the kernel's draw order."""
    checked = 0
    cases = [
        *(
            (HIDDEN, T, sm)
            for T in (64, 2049, 4096, 8192, 14928, 15566, 16114, 16263, 32768, 65536)
            for sm in (1, 7, 148, 216)
        ),
        *(
            (hidden, T, sm)
            for hidden in NEW_HIDDEN_SIZES
            for T in (2049, 15566, 24576)
            for sm in (7, 148)
        ),
    ]
    for hidden, T, sm in cases:
        P0, _, _, rpc0 = nd.k7_chunks(T, sm, hidden=hidden)
        rt = rpc0 // 64
        geometries = [(P0, rpc0, P0, rpc0)]
        for rb in range(max(1, rt - 6), rt + 1):
            for S in range(1, 6):
                geometry = nd._lpt_geometry(T, rb, S)
                if geometry is not None:
                    geometries.append(geometry)
        for geometry in geometries:
            units = _units(T, geometry, hidden)
            runs = nd._lpt_units(T, geometry, hidden=hidden)
            assert [t for count, t in runs for _ in range(count)] == units, (
                hidden,
                T,
                geometry,
            )
            for o_tiles in (1.0, 0.3):
                got = nd._lpt_makespan(runs, sm, o_tiles)
                assert got == _heap_makespan(units, sm, o_tiles), (
                    hidden,
                    T,
                    sm,
                    geometry,
                    o_tiles,
                )
                checked += 1
    assert checked > 1000


@pytest.mark.cpu_only
@pytest.mark.parametrize("order", ("chunk", "slab"))
@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_forced_geometry_is_read_at_call_time(
    monkeypatch, hidden: int, order: str
) -> None:
    """``lpt_chunks`` and ``BWD_SLAB_ORDER`` are module globals read at call time, at every hidden size (on B200, whose
    routes hold the ``lpt`` record of each size, and its ``lpt_os`` copy where B200's ``BWD_SLAB_ORDER`` of the size
    has a ``"slab"`` band -- a size without one skips the ``"slab"`` case; the size's order set to ``order`` for every
    token count): a replacement that forces a geometry at one token count moves ``slab_plan``,
    ``bwd_slab_atomic_program``, the selection and the size's ``lpt`` (``lpt_os``) record's launch grid (``min(SMs, H
    / 64 x P)``) there, never the ``dynr`` (``dynr_os``) record's; a forced geometry that does not cover the tokens is
    a ``ValueError``; without the replacement the size's table decides (``None``: ``dynr`` / ``dynr_os`` and the
    one-size chunks).  Each of these asks the rule at the size (the replacement records the hidden size it is asked
    at): at 80 / 112 slabs both forced grids are the SM count, so only that record shows that ``launch_grid`` reads
    the record's hidden size."""
    arch, sm, T = "sm_100a", 148, 2049
    if order == "slab" and arch not in ATOMIC.get(_lpt_os(hidden), ()):
        pytest.skip(
            f'{arch} holds no {_lpt_os(hidden)}: no "slab" band of hidden {hidden}'
        )
    monkeypatch.setattr(nd, "sm_count", lambda _index: sm)
    monkeypatch.setattr(nd, "device_arch", lambda _index: arch)
    monkeypatch.setitem(nd.BWD_SLAB_ORDER[hidden], arch, ((None, order),))
    rule, forced, seen = nd.lpt_chunks, {}, []

    def forcing(t, s, a, hidden=HIDDEN):
        seen.append(hidden)
        return forced[t] if t in forced else rule(t, s, a, hidden=hidden)

    monkeypatch.setattr(nd, "lpt_chunks", forcing)
    suffix = "_os" if order == "slab" else ""
    slab, lpt = _record_of("dynr" + suffix, hidden), _record_of("lpt" + suffix, hidden)
    slabs = hidden // det.SLAB_D
    plan = nd.host_plan(T, sm, arch, hidden)
    one = (plan["parts"], plan["rows_per_chunk"], plan["parts"], plan["rows_per_chunk"])
    dynr_grid = (min(sm, slabs * plan["parts"]), 1, 1)
    # one small chunk (5 chunks); one round of units below the SM count at H 4096 (2 chunks: grid 128 on 148 SMs)
    for geometry in ((5, 448, 4, 320), (2, 1280, 1, 832)):
        forced[T] = geometry
        seen.clear()
        assert nd.slab_plan(T, sm, arch, hidden) == dict(
            zip(LPT_HOST_KEYS, geometry, strict=True)
        )
        assert nd.bwd_slab_atomic_program(T, sm, arch, hidden) == "lpt" + suffix
        names = nd.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
        assert names[-1] == lpt and slab not in names, names
        assert seen and set(seen) == {hidden}, seen
        seen.clear()
        assert nd.launch_grid(nd.MODULES[lpt], T, device_index=0) == (
            min(sm, slabs * geometry[0]),
            1,
            1,
        )
        assert seen == [hidden], seen
        assert nd.launch_grid(nd.MODULES[slab], T, device_index=0) == dynr_grid
        assert nd.slab_plan(T + 64, sm, arch, hidden) != dict(
            zip(LPT_HOST_KEYS, geometry, strict=True)
        )
    forced[T] = (2, 512, 1, 64)  # 576 tokens: not a partition of 2049
    with pytest.raises(ValueError, match="not a two-size geometry"):
        nd.slab_plan(T, sm, arch, hidden)
    with pytest.raises(ValueError, match="not a two-size geometry"):
        nd.launch_grid(nd.MODULES[lpt], T, device_index=0)
    del forced[T]
    assert (
        nd.bwd_slab_atomic_program(T, sm, arch, hidden)
        == ("lpt" if rule(T, sm, arch, hidden=hidden) else "dynr") + suffix
    )
    monkeypatch.setitem(_lpt_table(hidden), arch, None)
    assert nd.bwd_slab_atomic_program(T, sm, arch, hidden) == "dynr" + suffix
    names = nd.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
    assert names[-1] == slab and lpt not in names, names
    assert nd.slab_plan(T, sm, arch, hidden) == dict(
        zip(LPT_HOST_KEYS, one, strict=True)
    )
    assert nd.launch_grid(nd.MODULES[lpt], T, device_index=0) == dynr_grid


@pytest.mark.cpu_only
def test_workspace_envelope_never_runs_the_screen(monkeypatch) -> None:
    """The workspace sizes and the rules rows never run the column slab's screen: ``host_plan``, which the envelope
    reads at every 64-token tile, carries no ``lpt`` value.  With ``lpt_chunks``, ``_lpt_plan`` and
    ``bwd_slab_order`` replaced by functions that raise and the ``host_plan`` / envelope caches cleared, the backward
    envelope of 16384 tokens on B200 and the rules rows of record, at every hidden size, are their unpatched
    values."""
    arch, sm, rows = "sm_100a", 148, 16384
    want = {
        hidden: nd.workspace_bytes_upto("bwd", rows, sm, arch, hidden)
        for hidden in HIDDEN_SIZES
    }
    rows_of_record = {
        (hidden, T): nd.rules_row(T, sm, arch, hidden)
        for hidden in HIDDEN_SIZES
        for T in det.RULES_T_VALUES
    }

    def screen(*_args, **_kwargs):
        raise AssertionError(
            "the workspace or the rules ran the column slab's lpt screen"
        )

    monkeypatch.setattr(nd, "lpt_chunks", screen)
    monkeypatch.setattr(nd, "_lpt_plan", screen)
    monkeypatch.setattr(nd, "bwd_slab_order", screen)
    nd.workspace_bytes_upto.cache_clear()
    nd.host_plan.cache_clear()
    for hidden, nbytes in want.items():
        assert nd.workspace_bytes_upto("bwd", rows, sm, arch, hidden) == nbytes, hidden
    for (hidden, T), row in rows_of_record.items():
        assert nd.rules_row(T, sm, arch, hidden) == row, (hidden, T)


# (arch, sm, hidden): the boards and sizes other than 4096 whose architecture has no lpt record of the size (sm_103a:
# no calibration of these sizes yet; sm_107a: no route of these sizes).
_ONE_SIZE_CASES = [
    pytest.param(arch, sm, hidden, id=f"{arch}-{sm}-H{hidden}")
    for arch, sm in det.RULES_BOARDS
    for hidden in NEW_HIDDEN_SIZES
    if arch not in ATOMIC[_lpt(hidden)]
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "hidden"), _ONE_SIZE_CASES)
def test_column_slab_rule_without_a_calibration_of_the_size(
    arch: str, sm: int, hidden: int, monkeypatch
) -> None:
    """At hidden 5120 and 7168 on an architecture without the size's ``lpt`` record the size's table is ``None`` and
    the column slab runs its one-size ``dynr`` record (``dynr_os`` where the size's ``BWD_SLAB_ORDER`` band is
    ``"slab"``) at every token count.  With ``_lpt_plan`` replaced by a function that raises (the rule never runs the
    screen there), at the token counts of record, of ``LPT_SWEEP`` and of ``LPT_HIDDEN_T``: ``lpt_chunks`` is
    ``None`` and ``bwd_slab_atomic_program`` ``dynr`` / ``dynr_os``; ``slab_plan`` is the one-size chunks of
    ``k7_chunks`` at the size (``P`` and the rows per chunk of the deterministic ``host_plan`` of the size: both
    backwards sum the partials of the same chunks); where the architecture routes the size the ``bwd_finish``
    selection holds that record of the size."""

    def screen(*_args, **_kwargs):
        raise AssertionError(
            f"the column-slab rule of hidden {hidden} on {arch} ran the lpt screen"
        )

    assert _lpt_table(hidden)[arch] is None, (arch, hidden)
    monkeypatch.setattr(nd, "_lpt_plan", screen)
    checked = 0
    for T in sorted(set(T_VALUES) | set(LPT_SWEEP) | set(LPT_HIDDEN_T)):
        what = (arch, sm, hidden, T)
        assert nd.lpt_chunks(T, sm, arch, hidden=hidden) is None, what
        order = _band_at(T, nd.BWD_SLAB_ORDER[hidden][arch])
        program = "dynr" + ("_os" if order == "slab" else "")
        assert nd.bwd_slab_atomic_program(T, sm, arch, hidden) == program, what
        plan = det.host_plan(T, sm, arch, hidden)
        parts, _units, _grid, rows_per_chunk = nd.k7_chunks(T, sm, hidden=hidden)
        assert (parts, rows_per_chunk) == (
            plan["parts"],
            plan["rows_per_chunk"],
        ), what
        assert nd.slab_plan(T, sm, arch, hidden) == dict(
            zip(
                LPT_HOST_KEYS,
                (parts, rows_per_chunk, parts, rows_per_chunk),
                strict=True,
            )
        ), what
        if arch in NEW_HIDDEN_ARCHES:
            names = nd.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
            slabs = [n for n in names if nd.MODULES[n]["role"] == "bwd_slab"]
            assert slabs == [_record_of(program, hidden)], (what, names)
        checked += 1
    assert checked


def _valid_calibration(value) -> bool:
    """``None`` or ``(o_tiles, keep)`` with ``o_tiles > 0`` and ``0 < keep <= 1``."""
    return value is None or (len(value) == 2 and value[0] > 0 and 0 < value[1] <= 1)


def _valid_bands(bands: tuple, valid) -> bool:
    """Token-count bands ``((t_upper, value), ..., (None, value))``: positive integer bounds strictly ascending, the
    unbounded band last and only last, every value ``valid``."""
    uppers = [t_upper for t_upper, _ in bands]
    return (
        bool(bands)
        and uppers[-1] is None
        and all(isinstance(t, int) and t > 0 for t in uppers[:-1])
        and uppers[:-1] == sorted(set(uppers[:-1]))
        and all(valid(value) for _, value in bands)
    )


# A banded calibration the rule is checked under (bands are a form of BWD_SLAB_LPT, not a table of record) and the
# token counts it is checked at: both sides of each band edge, below and above them.
_BANDS = ((33600, (0.5, 0.985)), (34368, (0.5, 0.98)), (None, (0.5, 0.97)))
_BAND_T = (1, 2049, 16263, 32768, 33599, 33600, 33601, 34368, 34369, 40000, 65536)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", (208, 212, 216))
def test_banded_lpt_calibration(sm: int) -> None:
    """``BWD_SLAB_LPT`` is the table of record (``LPT_OF_RECORD``); every entry of it and of ``BWD_SLAB_LPT_HIDDEN``
    is ``None``, one ``(o_tiles, keep)`` with ``o_tiles > 0`` and ``0 < keep <= 1``, or token-count bands of such
    values ending in an unbounded one.  With sm_107a's entry banded (``_BANDS``) at 208, 212 and 216 SMs the
    calibration of a token count is ``(0.5, 0.985)`` to 33600 tokens, ``(0.5, 0.98)`` to 34368 and ``(0.5, 0.97)``
    above (both sides of each edge), and ``lpt_chunks`` is the rule under the scalar table of that calibration; bands
    that leave a token count uncovered are a ``ValueError``."""
    assert nd.BWD_SLAB_LPT == LPT_OF_RECORD
    for hidden in HIDDEN_SIZES:
        for arch, entry in _lpt_table(hidden).items():
            if entry is not None and isinstance(entry[0], tuple):
                assert _valid_bands(entry, _valid_calibration), (hidden, arch, entry)
            else:
                assert _valid_calibration(entry), (hidden, arch, entry)
    arch, bands = "sm_107a", _BANDS
    assert _valid_bands(bands, _valid_calibration)
    want = {}
    with pytest.MonkeyPatch.context() as patch:
        _clear_plans()
        try:
            patch.setitem(nd.BWD_SLAB_LPT, arch, bands)
            assert [
                nd._lpt_calibration(T, arch) for T in (33600, 33601, 34368, 34369)
            ] == [
                (0.5, 0.985),
                (0.5, 0.98),
                (0.5, 0.98),
                (0.5, 0.97),
            ]
            for T in _BAND_T:
                patch.setitem(nd.BWD_SLAB_LPT, arch, _calibration_at(bands, T))
                want[T] = nd.lpt_chunks(T, sm, arch)
            patch.setitem(nd.BWD_SLAB_LPT, arch, bands)
            for T in _BAND_T:
                assert nd._lpt_calibration(T, arch) == _calibration_at(bands, T), T
                assert nd.lpt_chunks(T, sm, arch) == want[T], T
            patch.setitem(nd.BWD_SLAB_LPT, arch, ((33600, (0.5, 0.985)),))
            with pytest.raises(ValueError, match="uncovered"):
                nd.lpt_chunks(33601, sm, arch)
        finally:
            _clear_plans()


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_slab_order_bands(arch: str, sm: int) -> None:
    """``BWD_SLAB_ORDER`` is the table of record (``ORDER``): an entry per hidden size and architecture, token-count
    bands ending in an unbounded one, each order ``"chunk"`` or ``"slab"``; ``bwd_slab_order`` is the band's order on
    both sides of every band edge, and another order or bands that leave a token count uncovered are a
    ``ValueError``.  The table is read at call time: with the bands ``((2049, "slab"), (None, "chunk"))`` at every
    hidden size of the board's architecture, 2049 tokens select the ``_os`` copy of the form ``lpt_chunks`` picks
    where the architecture has that record of the size (where it has none, as on sm_107a, the ``"slab"`` band is a
    ``ValueError`` of the selection, never a backward without the column slab) and 2050 tokens the form itself."""
    assert nd.BWD_SLAB_ORDER == ORDER
    for hidden, per_arch in nd.BWD_SLAB_ORDER.items():
        assert set(per_arch) == set(det.ARCHES), hidden
        for a, bands in per_arch.items():
            assert _valid_bands(bands, lambda order: order in ("chunk", "slab")), (
                hidden,
                a,
                bands,
            )
            for t_upper, _order in bands:
                for T in (1,) if t_upper is None else (t_upper, t_upper + 1):
                    assert nd.bwd_slab_order(T, a, hidden) == _band_at(T, bands), (
                        hidden,
                        a,
                        T,
                    )
    assert set(nd.BWD_SLAB_ORDER) == set(HIDDEN_SIZES)
    with pytest.MonkeyPatch.context() as patch:
        _clear_plans()
        try:
            for hidden in _arch_sizes(arch):
                bands = ((2049, "slab"), (None, "chunk"))
                patch.setitem(nd.BWD_SLAB_ORDER[hidden], arch, bands)
                for T, suffix in ((2049, "_os"), (2050, "")):
                    what = (hidden, T)
                    geometry = nd.lpt_chunks(T, sm, arch, hidden=hidden)
                    program = nd.bwd_slab_atomic_program(T, sm, arch, hidden)
                    assert program == ("lpt" if geometry else "dynr") + suffix, what
                    if arch not in ATOMIC.get(_record_of(program, hidden), ()):
                        assert suffix, what
                        with pytest.raises(
                            ValueError, match="selects 0 column-slab records"
                        ):
                            nd.selected_modules(
                                arch, hidden, "bwd_finish", rows=T, sm_count=sm
                            )
                        continue
                    names = nd.selected_modules(
                        arch, hidden, "bwd_finish", rows=T, sm_count=sm
                    )
                    slabs = [n for n in names if nd.MODULES[n]["role"] == "bwd_slab"]
                    assert slabs == [_record_of(program, hidden)], (what, names)
            patch.setitem(nd.BWD_SLAB_ORDER[HIDDEN], arch, ((2049, "slab"),))
            with pytest.raises(ValueError, match="uncovered"):
                nd.bwd_slab_order(2050, arch)
            patch.setitem(nd.BWD_SLAB_ORDER[HIDDEN], arch, ((None, "slabs"),))
            with pytest.raises(ValueError, match="unit order"):
                nd.bwd_slab_order(2050, arch)
        finally:
            _clear_plans()


@pytest.mark.cpu_only
def test_route_glue_and_supported_hidden_sizes() -> None:
    """The hidden sizes of each architecture are the deterministic package's: 4096, 5120 and 7168 on sm_100a /
    sm_103a, 4096 on sm_107a, none on another architecture; a route applies exactly where the deterministic one
    does; a size or architecture without a route is a ``ValueError`` of the selection, and a size the package does
    not list one of the column slab's rules (``lpt_chunks``, ``bwd_slab_order``, ``bwd_slab_atomic_program``,
    ``slab_plan``; as in ``host_plan``)."""
    for arch in det.ARCHES:
        sizes = nd.supported_hidden_sizes(arch)
        assert sizes == _arch_sizes(arch), arch
        assert sizes == (HIDDEN_SIZES if arch in NEW_HIDDEN_ARCHES else (HIDDEN,)), arch
        # the sizes the architecture does not route (sm_107a: 5120 and 7168)
        for hidden in (2048, 6144, *(h for h in HIDDEN_SIZES if h not in sizes)):
            with pytest.raises(ValueError):
                nd.selected_modules(arch, hidden, "bwd_finish", rows=129, sm_count=148)
        for bad in (2048, 6144):
            with pytest.raises(ValueError, match="hidden"):
                nd.lpt_chunks(129, 148, arch, hidden=bad)
            with pytest.raises(ValueError, match="hidden"):
                nd.bwd_slab_order(129, arch, bad)
            with pytest.raises(ValueError, match="hidden"):
                nd.bwd_slab_atomic_program(129, 148, arch, bad)
            with pytest.raises(ValueError, match="hidden"):
                nd.slab_plan(129, 148, arch, bad)
    assert nd.supported_hidden_sizes("sm_90a") == ()
    for kind in det.KINDS:
        for hidden in HIDDEN_SIZES:
            assert nd.route_key(hidden, kind) == det.route_key(hidden, kind)
        for capability in ((10, 0), (10, 3), (10, 7)):
            arch = nd.arch_for_capability(capability)
            for hidden in (2048, 6144, *HIDDEN_SIZES):
                applies = nd.route_applies(
                    device_capability=capability, hidden=hidden, kind=kind
                )
                assert applies == det.route_applies(
                    device_capability=capability, hidden=hidden, kind=kind
                ), (capability, hidden, kind)
                assert applies == (hidden in _arch_sizes(arch)), (capability, hidden)
        for capability in ((9, 0), (12, 0), (10, 1)):
            for hidden in HIDDEN_SIZES:
                assert not nd.route_applies(
                    device_capability=capability, hidden=hidden, kind=kind
                )
    with pytest.raises(ValueError):
        nd.route_key(HIDDEN, "bwd_reduce")
    with pytest.raises(ValueError):
        nd.selected_modules("sm_90a", HIDDEN, "bwd_finish", rows=129, sm_count=148)
    assert pkg.route_modules is nd.route_modules and pkg.route_key is nd.route_key


@pytest.mark.cpu_only
def test_library_names_and_record_scalars() -> None:
    """``library_name`` is ``<library>_<arch>`` on the record's architectures and a ``RuntimeError`` elsewhere;
    ``RECORD_SCALARS`` is the deterministic table of every hidden size renamed without the backward fn splits (the
    forward fn split of each size ``want_t = 0``; the atomic fn splits take no ``want_t``)."""
    for name, record in nd.MODULES.items():
        for arch in det.ARCHES:
            if arch in record["arches"]:
                assert nd.library_name(name, arch) == f"{record['library']}_{arch}"
            else:
                with pytest.raises(RuntimeError, match=f"not delivered for {arch}"):
                    nd.library_name(name, arch)
    assert {
        _renamed(name): scalars
        for name, scalars in DET_RECORD_SCALARS.items()
        if _det_role(name) != "fn_split_bwd"
    } == nd.RECORD_SCALARS
    assert {
        ND_PREFIX + "fn_split" + _suffix(hidden): {"want_t": 0}
        for hidden in HIDDEN_SIZES
    } == nd.RECORD_SCALARS


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
        f"{arch}/{sm}/{hidden}": list(jit.selected_modules(arch, hidden, "bwd_finish", rows=4096, sm_count=sm))
        for arch, sm in jit.RULES_BOARDS
        for hidden in jit.supported_hidden_sizes(arch)
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
    sources) imports, plans its routes (every hidden size of every board) and refuses ``deterministic=True`` in an
    interpreter where every ``flashinfer`` import raises; no flashinfer module is loaded."""
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
    boards = [
        (arch, sm, hidden)
        for arch, sm in det.RULES_BOARDS
        for hidden in _arch_sizes(arch)
    ]
    assert set(out["selection"]) == {f"{a}/{s}/{h}" for a, s, h in boards}
    for arch, sm, hidden in boards:
        assert out["selection"][f"{arch}/{sm}/{hidden}"] == list(
            nd.selected_modules(arch, hidden, "bwd_finish", rows=4096, sm_count=sm)
        ), (arch, sm, hidden)
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
