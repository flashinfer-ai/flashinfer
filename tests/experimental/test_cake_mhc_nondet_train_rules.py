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
package ``flashinfer.experimental.cake.cake_mhc_nondet_train`` (FlashInfer
issue #8068; no GPU needed, nothing is built or launched).

The package is a standalone copy of the deterministic package
(``flashinfer.experimental.cake.cake_mhc_det_train``) under the ``cake_mhc_train_``
prefix at its own hidden sizes (4096 on every architecture, 5120 and 7168 on
sm_100a / sm_103a: ``ND_HIDDEN_SIZES``), except the ``bwd_finish`` routes.  The
deterministic package also exports 5120 and 7168 on sm_107a, which this package
does not copy.  This module checks the registry ``cake_jit.py`` against the
deterministic registry's records, routes and record scalars of this package's
(architecture, hidden size) pairs on its boards of record
(``RULES_BOARDS``) and token counts of record (``RULES_T_VALUES``):

* the program set is the deterministic set without its column-slab
  (``bwd_slab__*``) and final-reduction (``bwd_reduce__*``) records and its
  activation-checkpoint carry (role ``fn_split_carry``), renamed and otherwise
  equal field for field (the token owner's group-row forms ``_trows_dyn``
  included), plus the programs of this package: at hidden 4096
  ``cake_mhc_train_fn_split_atomic__tv128_dep`` on sm_100a / sm_103a,
  ``cake_mhc_train_fn_split_atomic__tv128_t128_dep1`` on sm_107a (its last CTA
  alone waits for the token-owner pass: ``SPLIT_CHAIN``),
  ``cake_mhc_train_bwd_slab_fold__dynr`` and its slab-fastest copy
  ``cake_mhc_train_bwd_slab_fold__dynr_os`` on all three; at 5120
  and 7168 ``cake_mhc_train_fn_split_atomic__tv128_dep__h<H>``,
  ``cake_mhc_train_bwd_slab_fold__dynr__h<H>`` and
  ``cake_mhc_train_bwd_slab_fold__dynr_os__h<H>`` on sm_100a / sm_103a (no
  two-size record, ``lpt`` / ``lpt_os``: no two-size calibration of record).  Their
  records keep the launch, grid and roles of the deterministic records they
  replace (the fn splits with the static PDL attribute: they follow the
  token-owner pass; the column slab with its fold arguments ``dfn_slot``,
  ``slab_cnt`` and ``prow``), and each ``_os`` record is its base record of the
  same size under its own name and program, on the architectures of its base
  record whose ``BWD_SLAB_ORDER`` of the size, or a board cell of
  ``BWD_SLAB_ORDER_OVERRIDES`` at the size, has a ``"slab"`` band (on sm_107a
  the cell of the 212-SM board alone);
* no record, route, record scalar or PDL edge names a deterministic library;
* every route and the selection on every route equal the deterministic ones,
  except ``bwd_finish``: the deterministic token-owner records (the group-row
  forms inside ``BWD_ROW_BANDS``, the run-time-count record at another Sinkhorn
  count), the atomic fn split of the architecture and size, the column slab's
  forms (one per call by ``bwd_slab_atomic_program``: ``dynr`` and
  ``dynr_os`` at hidden 4096 on every architecture and at 5120 and 7168 on
  sm_100a and sm_103a), no
  reduction; every program the column slab's tables and board cells can name
  has a record on the route, and every ``_os`` record a band of them that names
  it; the route plans bind the column slab's parameter rows (``prow`` and
  ``param_rows``: the group rows where the token owner writes them, else the
  per-CTA rows);
* the PDL bands without an SM count are the deterministic ones, but on sm_107a,
  whose column slab keeps the attribute on at hidden 4096 (a cell without a
  band); this package's own bands (``PDL_BAND_OVERRIDES``: B200 with 148 SMs,
  GB200 and GB300 with 152) replace them on their board alone, the most
  specific key first (the order of the keys under a test table), and the route
  plan passes the device's SM count;
* the workspaces: ``fwd`` the deterministic one; ``bwd`` the ticket region and
  the column slab's arrival counts ``slab_cnt`` first (a zero prefix of
  ``ZERO_PREFIX_BYTES`` at every token count and hidden size), then the
  deterministic regions without the trailing ``dfn_part`` (the parameter region
  the deterministic one: the group rows inside ``BWD_ROW_BANDS``), then the
  column slab's per-chunk ``dfn`` slots ``dfn_slot``; larger than the
  deterministic workspace at every token count;
* the host rules are copies (equal outputs, not the deterministic objects;
  the per-size tables the deterministic ones without the column-slab and
  reduction tables; ``host_plan`` the deterministic one plus the column slab's
  ``prow``, ``param_rows`` and ``slot_parts``), but this package's own
  column-slab rule (``BWD_SLAB_LPT``, ``BWD_SLAB_LPT_HIDDEN``,
  ``BWD_SLAB_ORDER``, ``BWD_SLAB_ORDER_OVERRIDES``, ``BWD_SLAB_LPT_OVERRIDES``,
  ``lpt_chunks``, ``bwd_slab_order``, ``bwd_slab_atomic_program``,
  ``slab_plan``): its geometries, its token-count bands (the unit order of
  record, and the banded calibration form under a test table), its board cells
  (those of record: the order cells of sm_107a with 212 SMs at 4096 and of sm_103a
  with 152 SMs at 7168, whose ``"slab"`` bands below 63092 hold column-slab
  chunk counts of at least 19; the exact key ``(arch, SM count, hidden)`` under
  a test table; every other board, the 212-SM board to 1876 tokens and the
  152-SM board outside those ``"slab"`` bands, selecting and planning as
  without them, also in the route plans) and
  its selection at every hidden size (no table of record calibrates an
  architecture, so every call runs the one-size chunks of ``k7_chunks``, the
  deterministic column slab's; the rule under the former calibrations and the
  kernel test's screen), its
  call-time lookup and a workspace envelope that never runs its screen;
* the package is standalone: no source file imports another flashinfer package
  (the shared ``cake_build.py`` JIT fallback aside, inside a function), and a
  copy under another name imports and plans with ``flashinfer`` blocked;
* the mode policy without a device: the mode is part of the names (every
  exported name but the ALL_CAPS constants carries the ``nondet`` token), no
  public callable has a ``deterministic`` parameter (the autograd
  ``Function.forward`` included) and a ``deterministic`` keyword is the
  interpreter's ``TypeError`` before any operand check; every entry point has
  the deterministic package's counterpart of the same verb; every output of
  the package repeats bit for bit on one device, so no call of the package
  raises PyTorch's alert of a non-deterministic operation, and under
  ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) every
  entry point refuses CPU operands with its operand error and no alert;
* the top-level module ``flashinfer.cake_mhc_nondet_train`` (read as source):
  every entry point forwards to this package's function of the same name with
  this package's parameters, the package is imported only on first use, and
  nothing is imported from the deterministic package;
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
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import PACKAGES, cake_package

KEY = "cake.cake_mhc_nondet_train"
PACKAGE_NAME = "flashinfer.experimental.cake.cake_mhc_nondet_train"
# The former import paths of this package: removed, with no alias.
FORMER_PACKAGES = (
    "flashinfer.experimental.cake.cake_mhc_train_backend",
    "flashinfer.cake_mhc_train_backend",
)
pkg = cake_package(KEY)
nd = cake_package(KEY, "cake_jit")
cake_build = cake_package(KEY, "cake_build")
det = cake_package("cake.cake_mhc_det_train", "cake_jit")
det_build = cake_package("cake.cake_mhc_det_train", "cake_build")

PACKAGE_DIR = Path(nd.__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
HIDDEN = 4096
# The package's hidden sizes: 4096 on every architecture, 5120 and 7168 on the architectures of NEW_HIDDEN_ARCHES
# (the deterministic package also exports 5120 and 7168 on sm_107a; this package does not).
HIDDEN_SIZES = (4096, 5120, 7168)
NEW_HIDDEN_SIZES = HIDDEN_SIZES[1:]
NEW_HIDDEN_ARCHES = ("sm_100a", "sm_103a")
# The hidden sizes this package exports per architecture: the (architecture, hidden size) pairs on which it is
# compared with the deterministic records, routes and record scalars (test_exported_hidden_sizes_per_architecture).
ND_HIDDEN_SIZES = {
    arch: HIDDEN_SIZES if arch in NEW_HIDDEN_ARCHES else (HIDDEN,)
    for arch in det.ARCHES
}
ND_PREFIX, DET_PREFIX = "cake_mhc_train_", "cake_mhc_det_train_"
# The programs of this package and their architectures (the library suffix ``__h<H>`` names a size other than 4096).
OWN = {
    "cake_mhc_train_fn_split_atomic__tv128_dep": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fn_split_atomic__tv128_t128_dep1": ("sm_107a",),
    "cake_mhc_train_bwd_slab_fold__dynr": ("sm_100a", "sm_103a", "sm_107a"),
    **{
        f"cake_mhc_train_{program}__h{hidden}": NEW_HIDDEN_ARCHES
        for hidden in NEW_HIDDEN_SIZES
        for program in ("fn_split_atomic__tv128_dep", "bwd_slab_fold__dynr")
    },
    # The slab-fastest copies of dynr, on the architectures whose BWD_SLAB_ORDER bands of the size, or a board cell of
    # BWD_SLAB_ORDER_OVERRIDES at the size (sm_107a: the 212-SM board's), hold a "slab" band. No two-size record
    # (lpt, lpt_os) ships: no two-size calibration of record is other than None (LPT_OF_RECORD), so no table can name
    # one.
    "cake_mhc_train_bwd_slab_fold__dynr_os": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_slab_fold__dynr_os__h5120": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab_fold__dynr_os__h7168": ("sm_100a", "sm_103a"),
}
# The (architecture, hidden size) pairs whose atomic fn split is the "dep1" form (only the last CTA waits for the
# token-owner pass); the "dep" form elsewhere.
SPLIT_CHAIN = {("sm_107a", HIDDEN): "dep1"}
SLAB = "cake_mhc_train_bwd_slab_fold__dynr"
LPT = "cake_mhc_train_bwd_slab_fold__lpt"
# The slab-fastest copies of the column slab's two forms (the units drawn slab-fastest; program key ``<form>_os``); the
# two-size names (LPT, LPT_OS) name no record of record.
SLAB_OS = "cake_mhc_train_bwd_slab_fold__dynr_os"
LPT_OS = "cake_mhc_train_bwd_slab_fold__lpt_os"
DET_DYNR = "cake_mhc_det_train_bwd_slab__dynr"
# The column slab's arguments beyond the deterministic dynr record's (which binds ``dfn_part`` instead), kernel
# parameter -> logical name: the outputs and the parameter rows it folds (``chunks``: the rows of ``param_partial``,
# bound to ``param_rows``), ``accumulate``, the per-chunk ``dfn`` slots, the arrival counts and the row layout.
FOLD_ARGS = {
    "dfn": "dfn",
    "param_partial": "param_partial",
    "dscale": "dscale",
    "dbase": "dbase",
    "chunks": "param_rows",
    "accumulate": "accumulate",
    "dfn_slot": "dfn_slot",
    "slab_cnt": "slab_cnt",
    "prow": "prow",
}
# The fold's arguments after the ticket (dynr) or after the lpt record's p_big / rpc_small, in kernel-parameter order.
FOLD_TAIL = [["buffer", "dfn_slot"], ["buffer", "slab_cnt"], ["parameter", "prow"]]
# The column slab's host-plan values beyond the deterministic plan's: the parameter rows it folds (``prow`` 1: the
# group rows [56, groups] of the ``_trows_dyn`` token owner, 0: the per-CTA rows [G6, 32]; ``param_rows`` their
# count) and the ``dfn`` slots of the workspace (``slot_parts = min(32, ceil(T / 64))``).
FOLD_KEYS = ("prow", "param_rows", "slot_parts")
# The column slab's arrival counts (one uint32 per column slab, at most H / 64 = 112) and the workspace prefix every
# call must find zero and leaves zero: the ticket region (256 bytes) and the arrival counts (512 bytes).
SLAB_CNT_WORDS = 128
ZERO_PREFIX_BYTES = 768
# The column slab's tables of record: the unit order per hidden size and architecture (BWD_SLAB_ORDER: on sm_100a
# "chunk" to 2047 / 5108 tokens at hidden 4096 / 7168 and "slab" above, "slab" at every token count at 5120; on
# sm_103a "chunk" to 13278 / 28825 / 63091 tokens at hidden 4096 / 5120 / 7168 and "slab" above; "chunk" at every size
# on sm_107a, which routes neither 5120 nor 7168) and the two-size calibration at hidden 4096 (BWD_SLAB_LPT: one
# (o_tiles, keep) or None per architecture: None on every one, so every call runs the deterministic one-size chunks of
# k7_chunks and dfn's per-chunk sums are the deterministic column slab's); the board cells of record replace both on
# their board alone (below).
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
LPT_OF_RECORD = {"sm_100a": None, "sm_103a": None, "sm_107a": None}
# The board cells of record (BWD_SLAB_ORDER_OVERRIDES / BWD_SLAB_LPT_OVERRIDES, keyed (arch, SM count, hidden)): R200
# with 212 SMs at hidden 4096 keeps the chunk order to 1876 tokens, then runs the slab order (no calibration cell: the
# two-size schedule is calibrated nowhere); GB300 (sm_103a with 152 SMs) at hidden 7168 has an
# order cell alone (no lpt record of the size there): the chunk order to 33792 tokens, then the slab order on 47 of the
# 48 token-count segments in 33793..63091 whose column-slab chunk count is at least 19 (GB300_CELL_MIN_P; GB300_CELL_TIE
# keeps the chunk order), its last band from 62465 running into the table's slab band from 63092.
ORDER_OVERRIDES = {
    ("sm_107a", 212, 4096): ((1876, "chunk"), (None, "slab")),
    ("sm_103a", 152, 7168): (
        (33792, "chunk"),
        (34048, "slab"),
        (35264, "chunk"),
        (35328, "slab"),
        (36096, "chunk"),
        (36480, "slab"),
        (36608, "chunk"),
        (36800, "slab"),
        (37376, "chunk"),
        (37696, "slab"),
        (37888, "chunk"),
        (38016, "slab"),
        (38144, "chunk"),
        (38272, "slab"),
        (38400, "chunk"),
        (38912, "slab"),
        (39424, "chunk"),
        (40128, "slab"),
        (40960, "chunk"),
        (41344, "slab"),
        (41984, "chunk"),
        (42688, "slab"),
        (43008, "chunk"),
        (43200, "slab"),
        (43264, "chunk"),
        (44160, "slab"),
        (44544, "chunk"),
        (44992, "slab"),
        (45312, "chunk"),
        (46208, "slab"),
        (46592, "chunk"),
        (46656, "slab"),
        (46848, "chunk"),
        (47424, "slab"),
        (48128, "chunk"),
        (48640, "slab"),
        (49152, "chunk"),
        (50112, "slab"),
        (50432, "chunk"),
        (51072, "slab"),
        (51200, "chunk"),
        (51584, "slab"),
        (51712, "chunk"),
        (52288, "slab"),
        (52480, "chunk"),
        (53568, "slab"),
        (54016, "chunk"),
        (54720, "slab"),
        (55040, "chunk"),
        (55936, "slab"),
        (56320, "chunk"),
        (57536, "slab"),
        (57600, "chunk"),
        (59584, "slab"),
        (59904, "chunk"),
        (60800, "slab"),
        (61184, "chunk"),
        (62208, "slab"),
        (62464, "chunk"),
        (None, "slab"),
    ),
}
LPT_OVERRIDES: dict = {}
# The GB300 order cell and the token counts it changes: BWD_SLAB_ORDER of sm_103a at 7168 runs the chunk order to
# 63091 tokens and the slab order above, so the cell differs from it in its "slab" bands in GB300_CELL_SPAN alone, at
# GB300_CELL_SLAB_T token counts, each with a column-slab chunk count (k7_chunks at 152 SMs and hidden 7168) of at
# least GB300_CELL_MIN_P: the chunk counts at which the slab order was measured faster there.
GB300_CELL = ("sm_103a", 152, 7168)
GB300_CELL_SPAN = (33793, 63091)
GB300_CELL_SLAB_T = 18355
GB300_CELL_MIN_P = 19
# The one segment of chunk count 19 in GB300_CELL_SPAN that keeps the chunk order (it read a tie on the second node).
GB300_CELL_TIE = (34817, 35264)
# The boards the column slab's board cells are checked on: the boards of record (the deterministic package's
# RULES_BOARDS, shared with it: 208 and 216 SMs on sm_107a; GB300, the board of the order cell at 7168, is one) and
# R200 with 212 SMs, the board of the cells of record at 4096.
SLAB_BOARDS = (*det.RULES_BOARDS, ("sm_107a", 212))
# Both sides of every band edge of the board cells of record, and of the table band the GB300 cell runs into.
OVERRIDE_EDGES = tuple(
    sorted(
        {
            t + side
            for cells in (ORDER_OVERRIDES, LPT_OVERRIDES)
            for cell in cells.values()
            for t, _value in cell
            if t is not None
            for side in (0, 1)
        }
        | {GB300_CELL_SPAN[1], GB300_CELL_SPAN[1] + 1}
    )
)
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
# The 19 benchmark token counts of the former BWD_SLAB_LPT calibration, and per board the geometries the tables of
# record pick there: none (every one keeps the one-size form: dynr, or dynr_os where the board's unit-order band is
# "slab"). LPT_FORMER_FIRES: the geometries the screen picks there under the former calibrations (sm_100a (0.5, 0.99),
# sm_103a (0.5, 0.98); on sm_107a with 212 SMs (0.5, 0.985), the band of the board's former cell over 1877..33600):
# the rule itself is unchanged, only no table of record calibrates it.
BENCH_T = (1024, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 8192, 14928, 15288, 15566, 15590, 16114, 16132, 16164,
           16263, 32768, 65536)  # fmt: skip
LPT_FIRES = (("sm_100a", 148, {}), ("sm_103a", 152, {}), ("sm_107a", 212, {}))
LPT_FORMER_FIRES = (
    ("sm_100a", 148, (0.5, 0.99), {14928: (12, 1536, 9, 384), 15566: (12, 1600, 9, 448), 16114: (12, 1664, 9, 384)}),
    ("sm_103a", 152, (0.5, 0.98), {14928: (8, 2112, 7, 192), 15288: (10, 2048, 7, 320), 16263: (8, 2304, 7, 192)}),
    ("sm_107a", 212, (0.5, 0.985), {15566: (18, 1152, 13, 128), 16263: (17, 1216, 13, 128),
                                    32768: (14, 2496, 13, 320)}),
)  # fmt: skip
# At hidden 5120 and 7168, the token counts the two-size rule of the size is pinned at (the benchmark token counts and
# 24576), and per board and size the geometries the table of record (BWD_SLAB_LPT_HIDDEN) picks there: none.
# LPT_SCREEN_FIRES_HIDDEN: on B200 the geometries the kernel test's screen (1.0, 0.99) picks at token counts where the
# former calibrations of these sizes fired (H 5120 (1.0, 0.99), H 7168 (1.0, 0.98)).
LPT_HIDDEN_T = tuple(sorted({*BENCH_T, 24576}))
LPT_FIRES_HIDDEN = tuple(
    (arch, sm, hidden, {})
    for arch, sm in det.RULES_BOARDS
    for hidden in NEW_HIDDEN_SIZES
)
LPT_SCREEN_FIRES_HIDDEN = (
    ("sm_100a", 148, 5120, {15566: (13, 1408, 11, 64), 16263: (13, 1472, 11, 64), 32768: (13, 2816, 11, 896)}),
    ("sm_100a", 148, 7168, {2048: (7, 384, 5, 64), 15288: (10, 1664, 9, 320), 24576: (11, 2560, 9, 768)}),
)  # fmt: skip
# The deterministic programs this package does not copy: the column slab and the final reduction (the backward fn
# splits, whose libraries the forward does not share, have no copy either).
DROPPED = (DET_PREFIX + "bwd_slab__", DET_PREFIX + "bwd_reduce__")
DROPPED_ROLES = ("bwd_slab", "bwd_reduce")
# The deterministic activation-checkpoint carry, which this package does not copy either (its checkpointed backward
# keeps its atomic fn split): the role of the dual-layout fn split the carry launches, and the carry's table and rules.
CARRY_ROLES = ("fn_split_carry",)
CARRY_RULES = ("FN_SPLIT_CARRY", "fn_split_carry", "carried_modules")
# The deterministic group-row forms, which this package copies (its column slab folds the group rows where the token
# owner writes them): the token-owner records whose program ends in GROUP_ROWS (the reduction's "_trows" forms go with
# the dropped reduction), the per-size table and rule that select them (BWD_ROW_BANDS, bwd_group_rows), the constants
# of the group-row parameter region and the host-plan keys of that region.
GROUP_ROWS = "_trows_dyn"
DET_GROUP_ROWS = frozenset(
    name
    for name, record in det.MODULES.items()
    if (record["select_rule"] or {}).get("program", "").endswith(GROUP_ROWS)
)
GROUP_ROW_RULES = ("BWD_ROW_BANDS", "bwd_group_rows", "PARAM_ROW_COLS", "TOKEN_GROUP")
GROUP_ROW_KEYS = ("groups", "group_rows", "param_dim0", "param_dim1")
# Sinkhorn counts other than the count of record: the token-owner pass runs its run-time-count record.
RUN_TIME_ITERS = (1, 7, 64)
T_VALUES = tuple(T for T in det.RULES_T_VALUES if T > 0)
# The deterministic records, routes and record scalars of this package's (architecture, hidden size) pairs
# (ND_HIDDEN_SIZES): a record keeps the architectures that export its hidden size here; a record left with none (the
# deterministic sm_107a records at 5120 and 7168) has no copy in this package.
DET_MODULES = {
    name: {
        **record,
        "arches": [
            arch
            for arch in record["arches"]
            if record["hidden"] in ND_HIDDEN_SIZES.get(arch, ())
        ],
    }
    for name, record in det.MODULES.items()
    if any(
        record["hidden"] in ND_HIDDEN_SIZES.get(arch, ()) for arch in record["arches"]
    )
}
DET_ROUTES = {
    arch: {
        key: route
        for key, route in routes.items()
        if key
        in {
            det.route_key(hidden, kind)
            for hidden in ND_HIDDEN_SIZES.get(arch, ())
            for kind in det.KINDS
        }
    }
    for arch, routes in det.ROUTES.items()
}
DET_RECORD_SCALARS = {
    name: scalars for name, scalars in det.RECORD_SCALARS.items() if name in DET_MODULES
}
# The public entry points (every compute and workspace entry point, the autograd prepare / finish included).
ENTRIES = (
    "cake_mhc_nondet_train_forward_prepare",
    "cake_mhc_nondet_train_forward_stats",
    "cake_mhc_nondet_train_forward_finish",
    "cake_mhc_nondet_train_backward_prepare",
    "cake_mhc_nondet_train_backward_finish",
    "cake_mhc_nondet_train_prepare",
    "cake_mhc_nondet_train_finish",
    "cake_mhc_nondet_train_forward_workspace_bytes",
    "cake_mhc_nondet_train_backward_workspace_bytes",
    "cake_mhc_nondet_train_forward_workspace",
    "cake_mhc_nondet_train_backward_workspace",
)
# The other public functions.
HELPERS = (
    "is_cake_mhc_nondet_train_supported",
    "cake_mhc_nondet_train_reset_workspace",
    "cake_mhc_nondet_train_route_key",
    "cake_mhc_nondet_train_route_modules",
)
# The top-level module of the mode (read as source: the suite imports nothing but its package) and its entry points.
FACADE = "flashinfer/cake_mhc_nondet_train.py"
FACADE_ENTRIES = ("is_cake_mhc_nondet_train_supported", *ENTRIES)
# The prefixes of the public entry points of the two modes (the same verbs follow).
NONDET_API, DET_API = "cake_mhc_nondet_train", "cake_mhc_det_train"
# PyTorch's wording of the alert of a non-deterministic operation (no call of this package raises it).
_ALERT = "does not have a deterministic implementation"
# The interpreter's refusal of a ``deterministic`` keyword: no entry point has that parameter.
_NO_MODE_KEYWORD = "unexpected keyword argument 'deterministic'"
# The ``TypeError`` of ``cake_mhc_nondet_train_finish`` for a handle its prepare did not make.
_HANDLE_REFUSAL = (
    "handle must be the handle returned by cake_mhc_nondet_train_prepare (a cake_mhc_det_train_prepare "
    "handle goes to flashinfer.cake_mhc_det_train.cake_mhc_det_train_finish)"
)


def _mode_tokens(name: str) -> set[str]:
    """The mode words of ``name``: ``det`` / ``nondet`` as ``_``-separated tokens of a snake_case name, ``Det`` /
    ``Nondet`` as words of a CamelCase one (a substring test would read ``nondet`` as ``det`` too)."""
    if name[:1].isupper():
        return {word.lower() for word in re.findall(r"(Nondet|Det)(?=[A-Z]|$)", name)}
    return set(re.findall(r"(?:^|_)(det|nondet)(?=_|$)", name))


def _shape(parameters) -> list:
    """Name, kind and default of each parameter."""
    return [(p.name, p.kind, p.default) for p in parameters]


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
    (``fn_split_program(arch, 1, hidden)``), launched as the token-owner pass's programmatic dependent (``_dep``; on
    the (architecture, hidden size) pairs of ``SPLIT_CHAIN`` its ``_dep1`` form, whose last CTA alone waits)."""
    return (
        f"{ND_PREFIX}fn_split_atomic__{det.fn_split_program(arch, 1, hidden)}_"
        + SPLIT_CHAIN.get((arch, hidden), "dep")
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
        if arch in OWN.get(name, ())
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


def _order_cell(arch: str, sm: int | None, hidden: int = HIDDEN) -> tuple:
    """The unit-order bands a call on ``arch`` with ``sm`` SMs (``None``: no SM count) at ``hidden`` reads at call
    time: the cell of ``BWD_SLAB_ORDER_OVERRIDES`` under the exact key ``(arch, sm, hidden)``, else the size's
    ``BWD_SLAB_ORDER`` entry of the architecture."""
    key = (arch, sm, hidden)
    if sm is not None and key in nd.BWD_SLAB_ORDER_OVERRIDES:
        return nd.BWD_SLAB_ORDER_OVERRIDES[key]
    return nd.BWD_SLAB_ORDER[hidden][arch]


def _lpt_cell(arch: str, sm: int | None, hidden: int = HIDDEN):
    """The two-size calibration entry a call on ``arch`` with ``sm`` SMs (``None``: no SM count) at ``hidden`` reads at
    call time: the cell of ``BWD_SLAB_LPT_OVERRIDES`` under the exact key ``(arch, sm, hidden)``, else the size's
    table entry of the architecture (:func:`_lpt_table`)."""
    key = (arch, sm, hidden)
    if sm is not None and key in nd.BWD_SLAB_LPT_OVERRIDES:
        return nd.BWD_SLAB_LPT_OVERRIDES[key]
    return _lpt_table(hidden)[arch]


def _program_at(T: int, sm: int, arch: str, hidden: int = HIDDEN) -> str:
    """The column slab's program at ``T`` under the tables read at call time: ``lpt`` where ``lpt_chunks`` gives a
    geometry, else ``dynr``; ``_os`` appended where the board's unit-order band of ``T`` (:func:`_order_cell`) is
    ``"slab"``."""
    base = "lpt" if nd.lpt_chunks(T, sm, arch, hidden=hidden) else "dynr"
    order = _band_at(T, _order_cell(arch, sm, hidden))
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
    """The hidden sizes this package exports on ``arch`` (:data:`ND_HIDDEN_SIZES`; none on another architecture)."""
    return ND_HIDDEN_SIZES.get(arch, ())


def _det_role(name: str) -> str:
    return det.MODULES[name]["role"]


def _fold_plan(plan: dict) -> dict:
    """A deterministic host plan with the column slab's values (:data:`FOLD_KEYS`): ``prow`` 1 and ``param_rows`` the
    group count where the plan lays out the group rows (``group_rows``), else ``prow`` 0 and ``param_rows`` the
    token owner's CTA count ``G6`` (``chunks``); ``slot_parts = min(32, ceil(T / 64))``, at least the chunk count of
    the plan (``parts``) and of every two-size geometry."""
    group_rows = bool(plan["group_rows"])
    slot_parts = min(32, -(-plan["rows"] // det.SLAB_TILE_T))
    assert slot_parts >= plan["parts"], plan
    return {
        **plan,
        "prow": int(group_rows),
        "param_rows": plan["groups"] if group_rows else plan["chunks"],
        "slot_parts": slot_parts,
    }


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
    """``MODULES`` = every deterministic record of this package's (architecture, hidden size) pairs (``DET_MODULES``)
    but ``bwd_slab__*`` / ``bwd_reduce__*``, the backward fn splits and the checkpoint carry's records
    (``CARRY_ROLES``) under the ``cake_mhc_train_`` prefix
    (the record equal field for field after the rename: arches, hidden size, build knobs, launch, arguments, rules;
    the token owner's group-row forms ``DET_GROUP_ROWS`` included, and every token owner selects by the
    deterministic rule, so none by its ``R`` regimes alone) plus the programs of this package (``OWN``); the dropped
    records are exactly the deterministic column-slab and reduction roles, and the backward fn splits and the carry
    records are the only other records without a copy (no record of this package has a carry role, and the carry's
    table and rules, ``CARRY_RULES``, have no copy)."""
    assert tuple(det.HIDDEN_SIZES) == HIDDEN_SIZES
    assert {record["hidden"] for record in det.MODULES.values()} == set(HIDDEN_SIZES)
    assert {record["hidden"] for record in DET_MODULES.values()} == set(HIDDEN_SIZES)
    # every deterministic record at the architectures that export its size in this package; a record with none of
    # them (sm_107a at 5120 and 7168) has no copy
    for name, record in det.MODULES.items():
        arches = [a for a in record["arches"] if record["hidden"] in _arch_sizes(a)]
        want = {**record, "arches": arches} if arches else None
        assert DET_MODULES.get(name) == want, name
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
    carries = {name for name in DET_MODULES if _det_role(name) in CARRY_ROLES}
    assert carries and not carries & dropped
    assert DET_GROUP_ROWS and {_det_role(name) for name in DET_GROUP_ROWS} == {
        "bwd_token"
    }
    copied = {
        _renamed(name): name
        for name in DET_MODULES
        if name not in dropped and name not in backward_splits and name not in carries
    }
    assert {_renamed(name) for name in DET_GROUP_ROWS} <= set(copied)
    assert all(name.startswith(ND_PREFIX) for name in copied)
    assert not set(copied) & set(OWN)
    expected = set(copied) | set(OWN)
    assert set(nd.MODULES) == expected, sorted(set(nd.MODULES) ^ expected)
    assert not [
        name for name, record in nd.MODULES.items() if record["role"] in CARRY_ROLES
    ]
    assert not [
        name
        for name, record in nd.MODULES.items()
        if (record["select_rule"] or {}).get("kind") == "chunk_rows_band"
    ]
    for name in CARRY_RULES:
        assert hasattr(det, name) and not hasattr(nd, name), name
    for name, det_name in copied.items():
        assert nd.MODULES[name] == _renamed(DET_MODULES[det_name]), name
    for name, record in nd.MODULES.items():
        assert record["kernel_symbol"] == f"kernel_{record['kernel']}", name
        assert record["library"] == record["kernel"], name
        assert record["library"].startswith(ND_PREFIX), name
        assert record["library"].endswith(_suffix(record["hidden"])), name


@pytest.mark.cpu_only
def test_own_records_keep_the_deterministic_launch() -> None:
    """The programs of this package, at every hidden size: named after their library and kernel symbol (``__h<H>``
    at 5120 and 7168); the fn splits keep the launch and fixed grid of the deterministic transposed-layout split of
    their form and size but for the static PDL attribute (``use_pdl`` True: they follow the token-owner pass and wait
    for it) and bind ``fn``, ``fnTp``, ``dfn`` and ``accumulate`` (no ``fnp``, no ``want_t``); the column slab keeps
    the launch and grid rule of the deterministic ``dynr`` record of its size (``slabs`` = H / 64), binds its
    arguments without ``dfn_part`` and adds :data:`FOLD_ARGS` (``dfn``, ``param_partial``, ``dscale``, ``dbase``, the
    parameter row count ``chunks`` bound to ``param_rows``, ``accumulate`` and, right after the ticket, the per-chunk
    ``dfn`` slots ``dfn_slot``, the arrival counts ``slab_cnt`` and the row layout ``prow``).  At every hidden size
    the column slab's records select by ``bwd_slab_atomic_program`` (programs ``dynr`` / ``lpt`` / ``dynr_os`` /
    ``lpt_os``: the last ``__`` part of the name without the size suffix; on sm_103a at 5120 and 7168, whose route
    holds the ``dynr`` and ``dynr_os`` records, the rule names one of these at every token count, and on sm_107a the
    ``_os`` ones only on the board of its cell of ``BWD_SLAB_ORDER_OVERRIDES``); the ``lpt`` record of a size is its
    ``dynr`` record with the ``lpt_chunks`` chunk rule, its chunk scalars ``P`` / ``rpc`` bound to the ``lpt_*`` host
    values and ``p_big`` / ``rpc_small`` between the ticket and ``dfn_slot``, on the architectures the size's table
    calibrates (``BWD_SLAB_LPT`` at 4096, ``BWD_SLAB_LPT_HIDDEN`` at 5120 and 7168: a calibration without the record
    would select no column slab, a record without one would never run); the ``dynr_os`` / ``lpt_os`` record of a
    size, where the size has one, is its ``dynr`` / ``lpt`` record (the slab-fastest build has the same kernel
    parameters, launch and chunk rule) under its own name and program key, on the architectures of that record whose
    ``BWD_SLAB_ORDER`` of the size, or a board cell of ``BWD_SLAB_ORDER_OVERRIDES`` at the size, has a ``"slab"``
    band."""
    for name, arches in OWN.items():
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
        assert "dfn_part" in dynr["logical"] and not set(FOLD_ARGS) & set(
            dynr["logical"]
        ), hidden
        assert slab["logical"] == {
            **{k: v for k, v in dynr["logical"].items() if k != "dfn_part"},
            **FOLD_ARGS,
        }, hidden
        kept = [entry for entry in dynr["arg_plan"] if entry != ["buffer", "dfn_part"]]
        assert [
            entry for entry in slab["arg_plan"] if entry[1] not in FOLD_ARGS
        ] == kept
        assert slab["arg_plan"][-8] == ["buffer", "ticket"], hidden
        assert slab["arg_plan"][-7:-4] == FOLD_TAIL, hidden
    for hidden in HIDDEN_SIZES:
        table = _lpt_table(hidden)
        assert set(table) == set(det.ARCHES), hidden
        calibrated = {
            arch for arch, calibration in table.items() if calibration is not None
        }
        if _lpt(hidden) not in nd.MODULES:
            assert not calibrated and _lpt(hidden) not in OWN, hidden
            continue
        slab, lpt = nd.MODULES[_slab(hidden)], nd.MODULES[_lpt(hidden)]
        assert lpt["role"] == "bwd_slab" and lpt["hidden"] == hidden, hidden
        assert set(lpt["arches"]) <= set(slab["arches"]), hidden
        assert set(lpt["arches"]) == calibrated, hidden
        assert lpt["launch"] == slab["launch"], hidden
        assert lpt["grid_rule"] == {**slab["grid_rule"], "chunks_rule": "lpt_chunks"}
        assert lpt["arg_plan"] == [
            *slab["arg_plan"][:-7],
            ["parameter", "p_big"],
            ["parameter", "rpc_small"],
            *slab["arg_plan"][-7:],
        ], hidden
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
            if name not in OWN:
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
    without ``dfn_part``, then the column slab's workspace regions ``dfn_slot`` and ``slab_cnt``; ``LOGICAL_SCALARS``
    = the deterministic one, then the lpt record's chunk scalars and the column slab's ``param_rows`` and ``prow``;
    tensors as buffers, scalars as parameters), one logical name per kernel parameter, the per-call PDL argument
    exactly when ``use_pdl`` is ``"runtime"`` (just before the grid).  The host binds every
    argument of every routed record: each logical name is a call value, a workspace region, a host-plan scalar of
    the route's fast plan or a scalar the record fixes (the run-time-count records also take ``sinkhorn_iters``,
    which the full path passes), and every workspace value names a region of the route's workspace."""
    assert list(nd.LOGICAL_TENSORS) == [
        *(t for t in det.LOGICAL_TENSORS if t != "dfn_part"),
        "dfn_slot",
        "slab_cnt",
    ]
    assert "dfn_part" in det.LOGICAL_TENSORS
    assert set(GROUP_ROW_KEYS) & set(det.LOGICAL_SCALARS) == {"groups"}
    assert (
        tuple(det.LOGICAL_SCALARS) + LPT_HOST_KEYS + ("param_rows", "prow")
        == nd.LOGICAL_SCALARS
    )
    assert set(pkg._HOST_VALUES["bwd_finish"]) >= {*LPT_HOST_KEYS, "param_rows", "prow"}
    assert {"dfn_slot", "slab_cnt"} <= {
        value for value, _ in pkg._WORKSPACE_VALUES["bwd_finish"]
    }
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
    token-owner records (renamed, in route order: the group-row forms ``_trows_dyn`` where the deterministic route
    holds them), the atomic fn split of the architecture and size and the column slab's forms of the size (``dynr``,
    ``lpt``, ``dynr_os``, ``lpt_os``; on sm_103a at 5120 and 7168 ``dynr`` and ``dynr_os``).  Every record is routed,
    on exactly its architectures."""
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
    ``dynr`` record, as its ``_os`` copy where the board's unit-order band is ``"slab"``: :func:`_program_at`) -- one
    fn split, one column slab, no reduction.  The token owner is the group-row form (``GROUP_ROWS``) of its program
    where ``bwd_group_rows`` holds at the count of record, the run-time-count record at another count, the per-CTA
    form elsewhere."""
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
                rule = nd.MODULES[tokens[0]]["select_rule"] or {}
                assert (rule.get("kind") == "run_time_iters") == (
                    iters != det.SINKHORN_ITERS
                ), what
                group_rows = (
                    det.bwd_group_rows(T, sm, arch, hidden)
                    and iters == det.SINKHORN_ITERS
                )
                assert rule.get("program", "").endswith(GROUP_ROWS) == group_rows, (
                    what,
                    tokens,
                )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("arch", "sm"), (*det.RULES_BOARDS, ("sm_100a", 152), ("sm_103a", 148))
)
def test_x_hint_band_edges_follow_the_deterministic_rule(arch: str, sm: int) -> None:
    """On both sides of every edge of the deterministic ``FWD_MAP_XNRM`` bands (the token counts where the mapping
    pass runs its program's ``_xn`` form, whose X loads carry the L2 hint EVICT_NORMAL: at hidden 4096 on sm_100a,
    sm_103a and sm_107a, none at 5120 / 7168), on the boards of record, GB200 (sm_100a with 152 SMs) and B300
    (sm_103a with 148 SMs), at every hidden size of the architecture: ``fwd_map_program`` is the deterministic
    program, an ``_xn`` form exactly inside a band of the size, and ``fwd_prepare`` / ``fwd_recompute`` select the
    renamed deterministic records, whose mapping-pass record runs an ``_xn`` library exactly there."""
    assert nd.FWD_MAP_XNRM == det.FWD_MAP_XNRM
    edges = {
        t
        for hidden in HIDDEN_SIZES
        for bands in det._tab("FWD_MAP_XNRM", hidden).values()
        for above, at_most in bands
        for t in (above, above + 1, at_most, at_most + 1)
    }
    assert edges
    for hidden in _arch_sizes(arch):
        bands = det._tab("FWD_MAP_XNRM", hidden)[arch]
        for T in sorted(t for t in edges if t > 0):
            what = (arch, sm, hidden, T)
            inside = any(above < T <= at_most for above, at_most in bands)
            program = nd.fwd_map_program(T, sm, arch, hidden)
            assert program == det.fwd_map_program(T, sm, arch, hidden), what
            assert program.endswith("_xn") == inside, (what, program)
            for kind in ("fwd_prepare", "fwd_recompute"):
                names = nd.selected_modules(arch, hidden, kind, rows=T, sm_count=sm)
                det_names = det.selected_modules(
                    arch, hidden, kind, rows=T, sm_count=sm
                )
                assert names == tuple(_renamed(n) for n in det_names), (what, kind)
                maps = [n for n in names if nd.MODULES[n]["role"] == "fwd_map"]
                assert maps and all(
                    nd.MODULES[n]["library"].endswith("_xn") == inside for n in maps
                ), (what, kind, maps)


def _possible_programs(arch: str, hidden: int) -> set[str]:
    """The programs ``bwd_slab_atomic_program`` can name on ``arch`` at ``hidden`` under the tables read at call
    time, on every board: the size's ``BWD_SLAB_ORDER`` and calibration entries (every SM count without a board cell)
    and, per SM count with a board cell of the size in ``BWD_SLAB_ORDER_OVERRIDES`` or ``BWD_SLAB_LPT_OVERRIDES``, the
    pair the board resolves (:func:`_order_cell`, :func:`_lpt_cell`; a board reads its two cells together, never one
    board's order with another's calibration).  Per pair, on each stretch of token counts where both its order band
    and its calibration band are constant, ``dynr`` (the screen may pick no geometry) and ``lpt`` where the
    calibration is not ``None``, each with ``_os`` where the order is ``"slab"``."""
    boards = {None} | {
        sm
        for cells in (nd.BWD_SLAB_ORDER_OVERRIDES, nd.BWD_SLAB_LPT_OVERRIDES)
        for a, sm, h in cells
        if (a, h) == (arch, hidden)
    }
    programs = set()
    for sm in boards:
        order, entry = _order_cell(arch, sm, hidden), _lpt_cell(arch, sm, hidden)
        bands = entry if entry is not None and isinstance(entry[0], tuple) else ()
        edges = sorted({t for t, _ in (*order, *bands) if t is not None})
        for T in (1, *(t + 1 for t in edges)):
            suffix = "_os" if _band_at(T, order) == "slab" else ""
            programs.add("dynr" + suffix)
            if _calibration_at(entry, T) is not None:
                programs.add("lpt" + suffix)
    return programs


@pytest.mark.cpu_only
def test_slab_tables_route_every_program_they_name() -> None:
    """On every architecture and routed hidden size: every program the column slab's tables can name (``BWD_SLAB_LPT`` /
    ``BWD_SLAB_LPT_HIDDEN`` with ``BWD_SLAB_ORDER``, and on each board with a cell of ``BWD_SLAB_ORDER_OVERRIDES`` /
    ``BWD_SLAB_LPT_OVERRIDES`` its two cells: :func:`_possible_programs`) has its record on the ``bwd_finish`` route
    (a program without one would select no column slab), one record per program; and every ``_os`` record on the
    route is one they can name: ``dynr_os`` needs a ``"slab"`` band of the size, ``lpt_os`` a calibration band other
    than ``None`` that overlaps one on the same board.  A size an architecture does not route has ``"chunk"`` bands
    alone (a ``"slab"`` band would name a slab-fastest program that has no record there)."""
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
    ``cake_mhc_train_bwd_slab_fold`` on ``K6->K7`` (every form and size); ``PDL_OFF_BANDS`` = the deterministic
    bands of every hidden size without ``K7->K8`` (one table: the deterministic package's per-size tables differ only
    in their K7->K8 cells; test_host_rules_are_copies_of_the_deterministic_ones).  On every path and
    token count (the counts of record and every band boundary) the ``enable_pdl`` without an SM count of the
    column-slab records of each hidden size reads the bands of :func:`_pdl_cell` without an SM count: every
    deterministic column-slab record's value of the route of that size where no ``(arch, None, hidden)`` cell holds
    the column slab's edge (the bands with an SM count: :func:`test_pdl_band_overrides_by_board`; the sm_107a cell:
    :func:`test_pdl_sm107a_column_slab_keeps_pdl_on`), the fn splits take no per-call argument (their attribute is
    static), and every copied record reads the deterministic value."""
    expected_edges = {
        _renamed(stem): edge
        for stem, edge in det.PDL_EDGES.items()
        if edge != "K7->K8" and not stem.endswith("bwd_slab")
    }
    expected_edges[ND_PREFIX + "bwd_slab_fold"] = "K6->K7"
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
            det_value = _pdl_value(
                T, det.PDL_OFF_BANDS.get(arch, {}).get("K6->K7", {}).get(path, ())
            )
            for hidden, (names, split, det_slabs) in slabs.items():
                value = _pdl_value(T, _pdl_cell(arch, None, hidden, "K6->K7", path))
                for name in names:
                    assert nd.enable_pdl(name, T, arch, path) == value, (T, path, name)
                for name in det_slabs:
                    assert det.enable_pdl(name, T, arch, path) == det_value, (
                        T,
                        path,
                        name,
                    )
                own = PDL_OVERRIDES.get((arch, None, hidden), {}).get("K6->K7", {})
                if own.get(path) is None:
                    assert value == det_value, (T, path, hidden)
                assert nd.enable_pdl(split, T, arch, path) == 1, (T, path, hidden)
            for name in copied:
                assert nd.enable_pdl(_renamed(name), T, arch, path) == det.enable_pdl(
                    name, T, arch, path
                ), (T, path, name)
    with pytest.raises(ValueError):
        nd.enable_pdl(SLAB, 1, arch, "sideways")


# This package's own PDL bands of record (``cake_jit.PDL_BAND_OVERRIDES``, measured on B200 with 148 SMs, on GB200
# with 152 SMs and on GB300 with 152 SMs).  B200: the collapse (``K1->K3``, forward) and the backward prepare
# (``K1->K5``, checkpointed backward) off in (2944, 3456] beside the deterministic (4096, 8192] at every hidden size,
# and the column slab (``K6->K7``, checkpointed backward) per hidden size.  GB200: the collapse and the backward
# prepare off in (2944, 3584] beside the deterministic (4096, 8192] at every hidden size, and the column slab per
# hidden size, open above at 7168.  GB300: the column slab (checkpointed backward) per hidden size, open above at 5120
# and 7168.  sm_107a (every SM count and none, hidden 4096): the column slab (checkpointed backward) without a band
# in place of the deterministic column slab's band.  Every other board (B300: sm_103a with 148 SMs) and every other
# edge of sm_107a keeps the deterministic bands.
PDL_OVERRIDES = {
    ("sm_100a", 148, None): {
        "K1->K3": {"forward": ((2944, 3456), (4096, 8192))},
        "K1->K5": {"checkpoint": ((2944, 3456), (4096, 8192))},
    },
    ("sm_100a", 148, 4096): {"K6->K7": {"checkpoint": ((1024, 2048), (4096, 8192))}},
    ("sm_100a", 148, 5120): {"K6->K7": {"checkpoint": ((1024, 2047),)}},
    ("sm_100a", 148, 7168): {"K6->K7": {"checkpoint": ((896, 8192),)}},
    ("sm_100a", 152, None): {
        "K1->K3": {"forward": ((2944, 3584), (4096, 8192))},
        "K1->K5": {"checkpoint": ((2944, 3584), (4096, 8192))},
    },
    ("sm_100a", 152, 4096): {"K6->K7": {"checkpoint": ((1024, 2159),)}},
    ("sm_100a", 152, 5120): {"K6->K7": {"checkpoint": ((1024, 1984), (7168, 8192))}},
    ("sm_100a", 152, 7168): {"K6->K7": {"checkpoint": ((768, None),)}},
    ("sm_103a", 152, 4096): {"K6->K7": {"checkpoint": ((1151, 2112), (4096, 8192))}},
    ("sm_103a", 152, 5120): {
        "K6->K7": {"checkpoint": ((1087, 1920), (4096, 8192), (24576, None))}
    },
    ("sm_103a", 152, 7168): {"K6->K7": {"checkpoint": ((831, None),)}},
    ("sm_107a", None, 4096): {"K6->K7": {"checkpoint": ()}},
}
# The boards with bands of their own (B200, GB200 and GB300; the sm_107a cell holds no SM count).
PDL_OVERRIDE_BOARDS = {
    (arch, sm) for arch, sm, _hidden in PDL_OVERRIDES if sm is not None
}
# The boards the per-call PDL argument is checked on: the boards of record, GB200 (sm_100a with 152 SMs) and B300
# (sm_103a with 148 SMs).
PDL_BOARDS = (*det.RULES_BOARDS, ("sm_100a", 152), ("sm_103a", 148))
# B200 and GB200 at the band edges and measured token counts: (hidden, or None for every size; edge; path; T; value
# with 148 SMs; value with 152 SMs; deterministic value).
PDL_POINTS = (
    *(
        (None, edge, path, T, b200, gb200, base)
        for edge, path in (("K1->K3", "forward"), ("K1->K5", "checkpoint"))
        for T, b200, gb200, base in ((2944, 1, 1, 1), (2945, 0, 0, 1), (3072, 0, 0, 1), (3456, 0, 0, 1),
                                     (3457, 1, 0, 1), (3584, 1, 0, 1), (3585, 1, 1, 1), (4096, 1, 1, 1),
                                     (4097, 0, 0, 0), (8192, 0, 0, 0), (8193, 1, 1, 1))
    ),
    (4096, "K6->K7", "checkpoint", 1024, 1, 1, 1), (4096, "K6->K7", "checkpoint", 2048, 0, 0, 0),
    (4096, "K6->K7", "checkpoint", 2049, 1, 0, 0), (4096, "K6->K7", "checkpoint", 2144, 1, 0, 0),
    (4096, "K6->K7", "checkpoint", 2159, 1, 0, 0), (4096, "K6->K7", "checkpoint", 2160, 1, 1, 0),
    (4096, "K6->K7", "checkpoint", 4096, 1, 1, 0),
    (4096, "K6->K7", "checkpoint", 4097, 0, 1, 0), (4096, "K6->K7", "checkpoint", 8192, 0, 1, 0),
    (4096, "K6->K7", "checkpoint", 8193, 1, 1, 1),
    (5120, "K6->K7", "checkpoint", 1984, 0, 0, 0), (5120, "K6->K7", "checkpoint", 1985, 0, 1, 0),
    (5120, "K6->K7", "checkpoint", 2047, 0, 1, 0), (5120, "K6->K7", "checkpoint", 2048, 1, 1, 0),
    (5120, "K6->K7", "checkpoint", 7168, 1, 1, 0), (5120, "K6->K7", "checkpoint", 7169, 1, 0, 0),
    (5120, "K6->K7", "checkpoint", 8192, 1, 0, 0), (5120, "K6->K7", "checkpoint", 8193, 1, 1, 1),
    (7168, "K6->K7", "checkpoint", 768, 1, 1, 1), (7168, "K6->K7", "checkpoint", 769, 1, 0, 1),
    (7168, "K6->K7", "checkpoint", 832, 1, 0, 1),
    (7168, "K6->K7", "checkpoint", 896, 1, 0, 1), (7168, "K6->K7", "checkpoint", 897, 0, 0, 1),
    (7168, "K6->K7", "checkpoint", 1024, 0, 0, 1), (7168, "K6->K7", "checkpoint", 8192, 0, 0, 0),
    (7168, "K6->K7", "checkpoint", 8193, 1, 0, 1), (7168, "K6->K7", "checkpoint", 65536, 1, 0, 1),
)  # fmt: skip
# GB300 at the column slab's band edges and measured token counts: (hidden, T, value with 152 SMs, deterministic
# value).
PDL_POINTS_GB300 = (
    (4096, 1025, 1, 1), (4096, 1151, 1, 1), (4096, 1152, 0, 1), (4096, 1536, 0, 1), (4096, 2112, 0, 1),
    (4096, 2113, 1, 1), (4096, 4096, 1, 1), (4096, 4097, 0, 0), (4096, 8192, 0, 0), (4096, 8193, 1, 1),
    (4096, 65536, 1, 1),
    (5120, 1025, 1, 1), (5120, 1087, 1, 1), (5120, 1088, 0, 1), (5120, 1920, 0, 1), (5120, 1921, 1, 1),
    (5120, 4097, 0, 0), (5120, 8193, 1, 1), (5120, 24576, 1, 1), (5120, 24577, 0, 1), (5120, 65536, 0, 1),
    (7168, 768, 1, 1), (7168, 831, 1, 1), (7168, 832, 0, 1), (7168, 4096, 0, 1), (7168, 8192, 0, 0),
    (7168, 8193, 0, 1), (7168, 32768, 0, 1), (7168, 65536, 0, 1),
)  # fmt: skip


def _pdl_cell(arch: str, sm: int | None, hidden: int, edge: str, path: str) -> tuple:
    """The bands of ``edge`` along ``path`` for a launch on ``arch`` with ``sm`` SMs (``None``: no SM count) at
    ``hidden``: the cell of the most specific key of :data:`PDL_OVERRIDES` that holds one (``(arch, sm, hidden)``,
    ``(arch, sm, None)``, ``(arch, None, hidden)``), else the deterministic cell."""
    keys = [(arch, None, hidden)]
    if sm is not None:
        keys = [(arch, sm, hidden), (arch, sm, None), *keys]
    for key in keys:
        cell = PDL_OVERRIDES.get(key, {}).get(edge, {}).get(path)
        if cell is not None:
            return cell
    return det.PDL_OFF_BANDS.get(arch, {}).get(edge, {}).get(path, ())


def _pdl_value(rows: int, bands) -> int:
    """0 when ``rows`` lies inside a band ``(above, at_most]`` of ``bands`` (``at_most`` ``None``: unbounded), 1
    elsewhere."""
    inside = any(
        above < rows and (at_most is None or rows <= at_most)
        for above, at_most in bands
    )
    return 0 if inside else 1


def _runtime_pdl_records(arch: str, hidden: int) -> list[str]:
    """The records of ``arch`` at ``hidden`` whose binding takes the per-call PDL argument."""
    return [
        name
        for name, record in nd.MODULES.items()
        if arch in record["arches"]
        and record["hidden"] == hidden
        and record["launch"]["use_pdl"] == "runtime"
    ]


def _pdl_token_counts(arch: str) -> list[int]:
    """The token counts of record, the benchmark token counts (:data:`LPT_HIDDEN_T`) and every boundary (``above``,
    ``above + 1``, ``at_most``, ``at_most + 1``) of the deterministic bands of ``arch`` and of every override band."""
    boundaries = set()
    for cells in (det.PDL_OFF_BANDS.get(arch, {}), *PDL_OVERRIDES.values()):
        for paths in cells.values():
            for bands in paths.values():
                for above, at_most in bands:
                    boundaries |= {above, above + 1}
                    if at_most is not None:
                        boundaries |= {at_most, at_most + 1}
    return sorted({*T_VALUES, *LPT_HIDDEN_T, *(t for t in boundaries if t > 0)})


@pytest.mark.cpu_only
def test_pdl_band_overrides_are_well_formed() -> None:
    """``PDL_BAND_OVERRIDES`` is the table of record (:data:`PDL_OVERRIDES`) and well formed: each key names a known
    architecture, an SM count (a positive int) or ``None`` and a hidden size the architecture serves or ``None``,
    never both ``None`` (that cell is ``PDL_OFF_BANDS``'s); each cell sits on an edge of ``PDL_EDGES`` and a path of
    ``PDL_PATHS`` and holds sorted, disjoint bands ``(above, at_most]`` of ints with ``0 <= above < at_most``, where
    only the last band may leave ``at_most`` ``None`` (no upper end), or no band where it keeps the attribute on over
    a band of ``PDL_OFF_BANDS``."""
    assert nd.PDL_BAND_OVERRIDES == PDL_OVERRIDES
    assert "PDL_BAND_OVERRIDES" in nd.__all__
    edges = set(nd.PDL_EDGES.values())
    for key, cells in nd.PDL_BAND_OVERRIDES.items():
        arch, sm, hidden = key
        assert arch in nd.ARCHES, key
        assert sm is None or (type(sm) is int and sm > 0), key
        assert hidden is None or hidden in nd.supported_hidden_sizes(arch), key
        assert (sm, hidden) != (None, None), key
        assert cells, key
        for edge, paths in cells.items():
            assert edge in edges and paths, (key, edge)
            for path, bands in paths.items():
                assert path in nd.PDL_PATHS, (key, edge, path)
                assert bands or nd.PDL_OFF_BANDS.get(arch, {}).get(edge, {}).get(
                    path
                ), (key, edge, path)
                last = None
                for i, (above, at_most) in enumerate(bands):
                    assert type(above) is int, (key, edge, bands)
                    if at_most is None:
                        assert i == len(bands) - 1, (key, edge, path, bands)
                    else:
                        assert type(at_most) is int, (key, edge, bands)
                    assert above >= 0, (key, edge, path, bands)
                    assert at_most is None or above < at_most, (key, edge, path, bands)
                    assert last is None or last <= above, (key, edge, path, bands)
                    last = at_most


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), PDL_BOARDS)
def test_pdl_band_overrides_by_board(arch: str, sm: int) -> None:
    """On every board (the boards of record, GB200 and B300), hidden size, path, record that takes the per-call PDL
    argument and token count (:func:`_pdl_token_counts`), ``enable_pdl`` with the board's SM count reads 0 exactly
    inside the bands of :func:`_pdl_cell` and 1 outside, and without an SM count the bands of :func:`_pdl_cell`
    without one: the deterministic bands but under an ``(arch, None, hidden)`` cell (the four-argument value, the
    deterministic record's for the copied records: the collapse and the backward prepare).  Only the boards with bands
    of their own (:data:`PDL_OVERRIDE_BOARDS`: B200 with 148 SMs, GB200 and GB300 with 152) read other values than
    the bands without an SM count; every other board reads the value without an SM count."""
    differs = 0
    for hidden in _arch_sizes(arch):
        names = _runtime_pdl_records(arch, hidden)
        assert names, (arch, hidden)
        for name in names:
            edge = nd.pdl_edge(nd.MODULES[name]["library"])
            assert edge in set(nd.PDL_EDGES.values()), name
            det_name = name.replace(ND_PREFIX, DET_PREFIX)
            copied = det_name in det.MODULES
            for path in nd.PDL_PATHS:
                cell = _pdl_cell(arch, sm, hidden, edge, path)
                base = _pdl_cell(arch, None, hidden, edge, path)
                assert (
                    base == det.PDL_OFF_BANDS.get(arch, {}).get(edge, {}).get(path, ())
                    or (arch, None, hidden) in PDL_OVERRIDES
                ), (hidden, name, path)
                for T in _pdl_token_counts(arch):
                    what = (hidden, name, path, T)
                    value = nd.enable_pdl(name, T, arch, path, sm_count=sm)
                    assert value == _pdl_value(T, cell), what
                    without = nd.enable_pdl(name, T, arch, path)
                    assert without == _pdl_value(T, base), what
                    if copied:
                        assert without == det.enable_pdl(det_name, T, arch, path), what
                    if (arch, sm) not in PDL_OVERRIDE_BOARDS:
                        assert value == without, what
                    differs += value != without
    assert (differs > 0) == ((arch, sm) in PDL_OVERRIDE_BOARDS), differs
    with pytest.raises(ValueError):
        nd.enable_pdl(SLAB, 1, arch, "sideways", sm_count=sm)


@pytest.mark.cpu_only
def test_pdl_band_overrides_at_the_measured_points() -> None:
    """On sm_100a the per-call argument at the band edges and measured token counts (:data:`PDL_POINTS`) of every
    record on the edge, at the point's hidden size (every size for the collapse and the backward prepare): with 148 SMs
    the B200 value, with 152 SMs the GB200 value and without an SM count the deterministic one."""
    arch = "sm_100a"
    for hidden in _arch_sizes(arch):
        by_edge: dict[str, list[str]] = {}
        for name in _runtime_pdl_records(arch, hidden):
            edge = nd.pdl_edge(nd.MODULES[name]["library"])
            by_edge.setdefault(edge, []).append(name)
        assert set(by_edge) == {"K1->K3", "K1->K5", "K6->K7"}, (hidden, by_edge)
        for size, edge, path, T, b200, gb200, base in PDL_POINTS:
            if size not in (None, hidden):
                continue
            for name in by_edge[edge]:
                what = (hidden, name, path, T)
                assert nd.enable_pdl(name, T, arch, path, sm_count=148) == b200, what
                assert nd.enable_pdl(name, T, arch, path, sm_count=152) == gb200, what
                assert nd.enable_pdl(name, T, arch, path) == base, what


@pytest.mark.cpu_only
def test_pdl_band_overrides_at_the_measured_points_gb300() -> None:
    """On GB300 the per-call argument at the column slab's band edges and measured token counts
    (:data:`PDL_POINTS_GB300`) of every column-slab record at the point's hidden size, on the checkpointed backward:
    with 152 SMs the GB300 value, with 148 SMs (B300) and without an SM count the deterministic one; the forward and
    the backward from saved statistics and the other edges keep the deterministic values."""
    arch = "sm_103a"
    for hidden in _arch_sizes(arch):
        slabs = [
            name
            for name in _runtime_pdl_records(arch, hidden)
            if nd.pdl_edge(nd.MODULES[name]["library"]) == "K6->K7"
        ]
        others = [
            name for name in _runtime_pdl_records(arch, hidden) if name not in slabs
        ]
        assert slabs and others, (hidden, slabs, others)
        points = [p for p in PDL_POINTS_GB300 if p[0] == hidden]
        assert points, hidden
        for _hidden, T, gb300, base in points:
            for name in slabs:
                what = (hidden, name, T)
                assert (
                    nd.enable_pdl(name, T, arch, "checkpoint", sm_count=152) == gb300
                ), what
                assert (
                    nd.enable_pdl(name, T, arch, "checkpoint", sm_count=148) == base
                ), what
                assert nd.enable_pdl(name, T, arch, "checkpoint") == base, what
                for path in ("forward", "backward"):
                    assert nd.enable_pdl(
                        name, T, arch, path, sm_count=152
                    ) == nd.enable_pdl(name, T, arch, path), what
            for name in others:
                for path in nd.PDL_PATHS:
                    assert nd.enable_pdl(
                        name, T, arch, path, sm_count=152
                    ) == nd.enable_pdl(name, T, arch, path), (hidden, name, path, T)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", (208, 212, 216, None))
def test_pdl_sm107a_column_slab_keeps_pdl_on(sm: int | None) -> None:
    """On sm_107a at hidden 4096 (its one size), with every R200 SM count and without one, every column-slab record
    of the ``bwd_finish`` route reads 1 on the checkpointed backward at the edges of the deterministic column slab's
    band (``PDL_OFF_BANDS``, ``K6->K7``, checkpointed backward), inside it, beside it and away from it, where the
    deterministic column slab reads 0 exactly inside that band; on the other paths the column slab, and on every path
    the other records that take the per-call argument (the collapse and the backward prepare), read the deterministic
    values."""
    arch = "sm_107a"
    assert _arch_sizes(arch) == (HIDDEN,)
    band = det.PDL_OFF_BANDS[arch]["K6->K7"]["checkpoint"]
    edges = {
        T for above, at_most in band for T in (above, above + 1, at_most, at_most + 1)
    }
    tokens = sorted(T for T in {1, 2048, 8192, *edges} if T > 0)
    assert any(_pdl_value(T, band) == 0 for T in tokens), band
    slabs = _slabs(arch)
    det_slabs = [
        n
        for n in det.route_modules(arch, HIDDEN, "bwd_finish")
        if _det_role(n) == "bwd_slab"
    ]
    others = [n for n in _runtime_pdl_records(arch, HIDDEN) if n not in slabs]
    assert slabs and det_slabs and others, (slabs, det_slabs, others)
    for T in tokens:
        det_value = _pdl_value(T, band)
        for name in det_slabs:
            assert det.enable_pdl(name, T, arch, "checkpoint") == det_value, (name, T)
        for name in slabs:
            what = (sm, name, T)
            assert nd.enable_pdl(name, T, arch, "checkpoint", sm_count=sm) == 1, what
            for path in ("forward", "backward"):
                assert nd.enable_pdl(
                    name, T, arch, path, sm_count=sm
                ) == det.enable_pdl(det_slabs[0], T, arch, path), (what, path)
        for name in others:
            det_name = name.replace(ND_PREFIX, DET_PREFIX)
            assert det_name in det.MODULES, name
            for path in nd.PDL_PATHS:
                assert nd.enable_pdl(
                    name, T, arch, path, sm_count=sm
                ) == det.enable_pdl(det_name, T, arch, path), (sm, name, T, path)


# A test table for the order of the keys: one band per key on the column slab's edge of sm_100a.  (sm_100a, 148, 5120)
# holds the forward cell alone, so its checkpoint launches take the next key's cell; no key names 7168 without an SM
# count, so those launches read PDL_OFF_BANDS ((1024, 8192] on the checkpointed backward, no band forward).
PDL_PRECEDENCE_TABLE = {
    ("sm_100a", 148, 4096): {"K6->K7": {"checkpoint": ((100, 200),)}},
    ("sm_100a", 148, None): {"K6->K7": {"checkpoint": ((300, 400),)}},
    ("sm_100a", None, 4096): {"K6->K7": {"checkpoint": ((500, 600),)}},
    ("sm_100a", None, 5120): {"K6->K7": {"checkpoint": ((500, 600),)}},
    ("sm_100a", 148, 5120): {"K6->K7": {"forward": ((700, 800),)}},
}
PDL_PRECEDENCE_T = (150, 350, 550, 750, 2000)
# (SM count, hidden size, path, the per-call argument at each of PDL_PRECEDENCE_T): the key that sets it.
PDL_PRECEDENCE = (
    (148, 4096, "checkpoint", (0, 1, 1, 1, 1)),  # (arch, sm, hidden) before (arch, sm, None) and (arch, None, hidden)
    (148, 5120, "checkpoint", (1, 0, 1, 1, 1)),  # (arch, sm, None) before (arch, None, hidden)
    (148, 5120, "forward", (1, 1, 1, 0, 1)),  # (arch, sm, hidden)
    (148, 7168, "checkpoint", (1, 0, 1, 1, 1)),  # (arch, sm, None)
    (152, 4096, "checkpoint", (1, 1, 0, 1, 1)),  # (arch, None, hidden) with an SM count
    (152, 7168, "checkpoint", (1, 1, 1, 1, 0)),  # PDL_OFF_BANDS
    (None, 4096, "checkpoint", (1, 1, 0, 1, 1)),  # (arch, None, hidden) without an SM count
    (None, 5120, "checkpoint", (1, 1, 0, 1, 1)),  # (arch, None, hidden), not (arch, 148, None)
    (None, 5120, "forward", (1, 1, 1, 1, 1)),  # PDL_OFF_BANDS, not (arch, 148, hidden)
    (None, 7168, "checkpoint", (1, 1, 1, 1, 0)),  # PDL_OFF_BANDS, not (arch, 148, None)
)  # fmt: skip


@pytest.mark.cpu_only
def test_pdl_band_override_precedence(monkeypatch) -> None:
    """The order of the keys of ``PDL_BAND_OVERRIDES`` under a test table (:data:`PDL_PRECEDENCE_TABLE`): per edge and
    path, ``(arch, sm_count, hidden)``, then ``(arch, sm_count, None)``, then ``(arch, None, hidden)``, then
    ``PDL_OFF_BANDS``; a key that holds only another cell of the edge leaves the cell to the next key, and a launch
    without an SM count matches no key with one.  The values of the ``dynr`` column-slab record of each hidden size on
    sm_100a are the literals of :data:`PDL_PRECEDENCE`."""
    arch = "sm_100a"
    names = {4096: SLAB, 5120: f"{SLAB}__h5120", 7168: f"{SLAB}__h7168"}
    monkeypatch.setattr(nd, "PDL_BAND_OVERRIDES", PDL_PRECEDENCE_TABLE)
    for sm, hidden, path, values in PDL_PRECEDENCE:
        name = names[hidden]
        assert nd.MODULES[name]["hidden"] == hidden, name
        got = tuple(
            nd.enable_pdl(name, T, arch, path, sm_count=sm) for T in PDL_PRECEDENCE_T
        )
        assert got == values, (sm, hidden, path, got)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", (148, 152))
def test_route_plan_passes_the_device_sm_count(monkeypatch, sm: int) -> None:
    """The route plan's per-call PDL argument is ``enable_pdl`` with the device's SM count: on sm_100a with ``sm`` SMs
    (the device's SM count, architecture and module loader replaced: nothing is built or launched), at every hidden
    size, kind, path and token count inside or beside the B200 bands, the launch of every record that takes the
    argument carries ``enable_pdl(name, T, arch, path, sm_count=sm)``, and no other launch carries one; with 148 SMs
    (B200) and with 152 SMs (GB200) some of these values differ from the value without an SM count."""
    arch = "sm_100a"

    class _Module:
        def __getattr__(self, entry: str) -> str:
            return entry

    monkeypatch.setattr(nd, "sm_count", lambda _index: sm)
    monkeypatch.setattr(nd, "device_arch", lambda _index: arch)
    monkeypatch.setattr(nd, "load", lambda _name, _arch: _Module())
    monkeypatch.setattr(pkg, "_PLANS", {})
    checked = differs = 0
    for hidden in _arch_sizes(arch):
        for T in (800, 1500, 3000, 6000):
            for kind in nd.KINDS:
                names = nd.selected_modules(arch, hidden, kind, rows=T, sm_count=sm)
                for path in nd.PDL_PATHS:
                    plan = pkg._route_plan(arch, kind, T, hidden, 0, path)
                    for name, launch in zip(names, plan, strict=True):
                        record = nd.MODULES[name]
                        positions = [
                            i
                            for i, (arg_kind, _) in enumerate(record["arg_plan"])
                            if arg_kind == "launch"
                        ]
                        if record["launch"]["use_pdl"] != "runtime":
                            assert not positions, name
                            continue
                        (position,) = positions
                        value = launch.template[position]
                        what = (hidden, T, kind, path, name)
                        assert value == nd.enable_pdl(
                            name, T, arch, path, sm_count=sm
                        ), what
                        differs += value != nd.enable_pdl(name, T, arch, path)
                        checked += 1
    assert checked
    assert differs > 0, differs


# ----------------------------------------------------------------------------------------------------------------
# Workspaces


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_workspace_holds_the_deterministic_regions_and_the_fold_regions(
    arch: str, sm: int
) -> None:
    """At every hidden size of the board's architecture: ``fwd`` is the deterministic forward workspace; ``bwd`` is the
    ticket region (offset 0), the column slab's arrival counts ``slab_cnt`` (``SLAB_CNT_WORDS`` uint32 at offset 256:
    with the ticket region the zero prefix of ``ZERO_PREFIX_BYTES`` at every token count and hidden size), the
    deterministic regions but the per-chunk partials ``dfn_part`` -- ``fnTp``, ``dzp``, ``tokp`` and the parameter
    region (the per-CTA partials ``[G6, 32]``, the group rows ``[56, ceil(T / 8)]`` inside ``BWD_ROW_BANDS``) -- with
    their deterministic sizes, dtypes and shapes, each ``ZERO_PREFIX_BYTES - 256`` bytes past its deterministic offset,
    and last the per-chunk ``dfn`` slot pairs ``dfn_slot`` (``2 x min(32, ceil(T / 64)) x 24 x 4H`` FP32, two planes of
    at least the deterministic ``P x 24 x 4H`` partials); the total is the deterministic total less the rounded
    ``dfn_part`` bytes plus the two regions, above the deterministic total at every token count; ``T == 0`` has no
    regions."""
    assert set(nd.WORKSPACE_REGIONS) == set(det.WORKSPACE_REGIONS) == {"fwd", "bwd"}
    assert nd.WORKSPACE_REGIONS["fwd"] == det.WORKSPACE_REGIONS["fwd"]
    det_bwd = det.WORKSPACE_REGIONS["bwd"]
    assert det_bwd[0]["name"] == "ticket" and det_bwd[-1]["name"] == "dfn_part"
    assert nd.SLAB_CNT_WORDS == SLAB_CNT_WORDS
    assert 4 * det.TICKET_WORDS + 4 * SLAB_CNT_WORDS == ZERO_PREFIX_BYTES
    assert max(hidden // det.SLAB_D for hidden in HIDDEN_SIZES) <= SLAB_CNT_WORDS
    assert nd.WORKSPACE_REGIONS["bwd"] == [
        det_bwd[0],
        {"dtype": "uint32", "name": "slab_cnt", "shape": [SLAB_CNT_WORDS]},
        *det_bwd[1:-1],
        {"dtype": "float32", "name": "dfn_slot", "shape": ["slot_parts", 24, "n"]},
    ]
    assert nd.WORKSPACE_OF_KIND == det.WORKSPACE_OF_KIND
    shift = ZERO_PREFIX_BYTES - 4 * det.TICKET_WORDS
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
        assert names[-1] == "dfn_part", what
        assert list(layout["regions"]) == [
            names[0],
            "slab_cnt",
            *names[1:-1],
            "dfn_slot",
        ], what
        assert layout["regions"]["ticket"] == reference["regions"]["ticket"], what
        assert layout["regions"]["ticket"]["offset"] == 0
        assert layout["regions"]["ticket"]["bytes"] == 4 * det.TICKET_WORDS
        assert layout["regions"]["slab_cnt"] == {
            "offset": 4 * det.TICKET_WORDS,
            "bytes": 4 * SLAB_CNT_WORDS,
            "dtype": "uint32",
            "shape": (SLAB_CNT_WORDS,),
        }, what
        for name in names[1:-1]:
            region = reference["regions"][name]
            assert layout["regions"][name] == {
                **region,
                "offset": region["offset"] + shift,
            }, (what, name)
        plan = det.host_plan(T, sm, arch, hidden)
        assert layout["regions"]["param_partial"]["shape"] == (
            (det.PARAM_ROW_COLS, plan["groups"])
            if plan["group_rows"]
            else (plan["chunks"], det.PARAM_COLS)
        ), what
        part = reference["regions"]["dfn_part"]
        parts = det.k7_chunks(T, sm, hidden=hidden)[0]
        assert part["shape"] == (parts, 24, 4 * hidden), what
        slots = layout["regions"]["dfn_slot"]
        slot_parts = min(32, -(-T // det.SLAB_TILE_T))
        assert slot_parts >= parts, what
        assert slots["offset"] == part["offset"] + shift, what
        assert slots["dtype"] == "float32", what
        assert slots["shape"] == (slot_parts, 24, 4 * hidden), what
        assert slots["bytes"] == slot_parts * 24 * 4 * hidden * 4, what
        assert layout["total_bytes"] == (
            reference["total_bytes"]
            - _round256(part["bytes"])
            + shift
            + _round256(slots["bytes"])
        ), what
        assert layout["total_bytes"] > reference["total_bytes"], what
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
    least the exact size of every token count up to ``T`` and above the deterministic envelope (a buffer of the
    deterministic sizing does not hold this package's layout); 0 at ``T == 0``; the forward envelope is the
    deterministic one.  Every token count below 4200 at hidden 4096, every seventh at 5120 and 7168."""
    for hidden in _arch_sizes(arch):
        previous = 0
        for T in range(1, 4200, 1 if hidden == HIDDEN else 7):
            what = (hidden, T)
            envelope = nd.workspace_bytes_upto("bwd", T, sm, arch, hidden)
            assert envelope >= previous, what
            assert envelope >= nd.workspace_bytes("bwd", T, sm, arch, hidden), what
            assert envelope > det.workspace_bytes_upto("bwd", T, sm, arch, hidden), what
            previous = envelope
        assert nd.workspace_bytes_upto("bwd", 0, sm, arch, hidden) == 0
        for rows in (64, 65, 1000, 2049, 8192, 16263, 20001, 65536):
            what = (hidden, rows)
            envelope = nd.workspace_bytes_upto("bwd", rows, sm, arch, hidden)
            assert envelope > det.workspace_bytes_upto("bwd", rows, sm, arch, hidden), (
                what
            )
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
    "GSTATS_COLS", "TOKP_COLS", "DZP_COLS", "PARAM_COLS", "PARAM_ROW_COLS", "TOKEN_GROUP", "TICKET_WORDS",
    "STATS_LAYOUT", "GSTATS_LAYOUT", "TOKP_LAYOUT", "PARAM_LAYOUT", "RULES_SM_COUNTS", "RULES_BOARDS",
    "RULES_T_VALUES", "PK_BANDS", "FN_SPLIT_PROGRAM", "FN_SPLIT_GRID", "FWD_MAP_PRODUCTION", "FWD_MAP_BANDS",
    "FWD_MAP_FIT", "FWD_MAP_CLAIM", "FWD_MAP_XNRM", "FWD_COEF_REGIMES", "BWD_TOKEN_PROGRAM", "BWD_TOKEN_REGIMES",
    "BWD_TOKEN_BANDS", "BWD_ROW_BANDS", "K5_GRID_BANDS", "K5_FORM_BANDS", "K5_FORM_ROWS", "FWD_FINISH_PROGRAM",
    "FWD_FINISH_ROWS", "FWD_FINISH_COL_SPLIT", "K3_TILING", "K6_CTAS_PER_SM", "PDL_PATHS", "SELECT_RULE_KINDS",
    "RUN_TIME_ITERS_ROLES",
)  # fmt: skip
# The deterministic column-slab and reduction rules, absent here (BWD_DPL_BANDS / BWD_DPL_SLABS: the bands and the
# column-slab programs of their L2-hint forms).
_DROPPED_RULES = (
    "BWD_SLAB_PROGRAM", "BWD_SLAB_BANDS", "BWD_SLAB_FIT", "BWD_REDUCE_PROGRAM", "BWD_REDUCE_REGIMES",
    "BWD_REDUCE_PARTS", "BWD_REDUCE_GRID", "BWD_DPL_BANDS", "BWD_DPL_SLABS", "bwd_slab_program",
    "bwd_reduce_program", "bwd_reduce_grid",
)  # fmt: skip
# This package's own column-slab rule, absent from the deterministic registry.
_OWN_RULES = (
    "BWD_SLAB_LPT",
    "BWD_SLAB_LPT_HIDDEN",
    "BWD_SLAB_ORDER",
    "BWD_SLAB_ORDER_OVERRIDES",
    "BWD_SLAB_LPT_OVERRIDES",
    "lpt_chunks",
    "bwd_slab_order",
    "bwd_slab_atomic_program",
    "slab_plan",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_host_rules_are_copies_of_the_deterministic_ones(arch: str, sm: int) -> None:
    """The constants and tables of the host rules equal the deterministic ones (the group-row table and constants,
    ``GROUP_ROW_RULES``, included); the rule functions are this package's own objects with the deterministic outputs
    on every token count of record (the host plan with the column slab's :data:`FOLD_KEYS` added:
    :func:`_fold_plan`); the deterministic column-slab and reduction rules are absent and ``PROGRAM_RULES`` drops
    them for this package's column-slab rule (``bwd_slab_atomic_program``, with ``BWD_SLAB_LPT``,
    ``BWD_SLAB_LPT_HIDDEN``, ``BWD_SLAB_ORDER``, ``BWD_SLAB_ORDER_OVERRIDES``, ``BWD_SLAB_LPT_OVERRIDES``,
    ``lpt_chunks``, ``bwd_slab_order`` and ``slab_plan``: none of them in the deterministic registry); a rules row is
    the deterministic row without its column-slab and reduction program columns, with this package's backward
    workspace bytes."""
    for name in _SHARED_TABLES:
        assert getattr(nd, name) == getattr(det, name), name
    # The hidden sizes: the deterministic ones; the per-size table names are the deterministic ones without the
    # column-slab and reduction tables (BWD_ROW_BANDS kept) and without PDL_OFF_BANDS, and the tables of each size
    # other than 4096 the deterministic tables of those names (the tables of record above are the H 4096 ones).
    # The deterministic PDL_OFF_BANDS of every size (4096 included) is this package's one table plus the K7->K8
    # cells, an edge this package does not have.
    assert tuple(nd.HIDDEN_SIZES) == tuple(det.HIDDEN_SIZES) == HIDDEN_SIZES
    dropped = (*_DROPPED_RULES, "PDL_OFF_BANDS")
    kept = tuple(name for name in det.H_TABLE_NAMES if name not in dropped)
    for hidden in HIDDEN_SIZES:
        without_k7_k8 = {
            a: {edge: bands for edge, bands in edges.items() if edge != "K7->K8"}
            for a, edges in det._tab("PDL_OFF_BANDS", hidden).items()
        }
        assert without_k7_k8 == nd.PDL_OFF_BANDS, hidden
    assert "BWD_ROW_BANDS" in kept
    assert kept == nd.H_TABLE_NAMES
    assert HIDDEN not in det.H_TABLES and set(det.H_TABLES) == set(NEW_HIDDEN_SIZES)
    assert {
        hidden: {name: det.H_TABLES[hidden][name] for name in nd.H_TABLE_NAMES}
        for hidden in NEW_HIDDEN_SIZES
    } == nd.H_TABLES
    for name in _DROPPED_RULES:
        assert hasattr(det, name) and not hasattr(nd, name), name
    for name in GROUP_ROW_RULES:
        assert hasattr(det, name) and hasattr(nd, name), name
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
        "bwd_group_rows",
        "bwd_token_program",
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
        assert nd.host_plan(T, sm, arch) == _fold_plan(det.host_plan(T, sm, arch)), T
        for rule in (
            "fwd_map_program",
            "fwd_coef_program",
            "bwd_group_rows",
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
    deterministic outputs (the host plan with the column slab's :data:`FOLD_KEYS`: :func:`_fold_plan`); the per-size
    tables are read through ``_tab`` (a size it does not spell is a ``KeyError``, never the H 4096 value; the column
    slab's ``BWD_SLAB_LPT`` / ``BWD_SLAB_LPT_HIDDEN`` / ``BWD_SLAB_ORDER`` and its board cells
    ``BWD_SLAB_ORDER_OVERRIDES`` / ``BWD_SLAB_LPT_OVERRIDES`` are no per-size tables of ``_tab``).
    ``rules_table()`` holds the H 4096 rows, then each other size's."""
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
            assert nd.host_plan(T, sm, arch, hidden) == _fold_plan(
                det.host_plan(T, sm, arch, hidden)
            ), what
            for rule in (
                "fwd_map_program",
                "fwd_coef_program",
                "bwd_group_rows",
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
    for name in (
        "BWD_SLAB_LPT",
        "BWD_SLAB_LPT_HIDDEN",
        "BWD_SLAB_ORDER",
        "BWD_SLAB_ORDER_OVERRIDES",
        "BWD_SLAB_LPT_OVERRIDES",
    ):
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
# BWD_SLAB_ORDER, BWD_SLAB_ORDER_OVERRIDES, BWD_SLAB_LPT_OVERRIDES, lpt_chunks, bwd_slab_order,
# bwd_slab_atomic_program, slab_plan)


def _clear_plans() -> None:
    """Drop the package's cached route plans (an experiment that edits ``BWD_SLAB_LPT`` / ``BWD_SLAB_LPT_HIDDEN`` /
    ``BWD_SLAB_ORDER`` / ``BWD_SLAB_ORDER_OVERRIDES`` / ``BWD_SLAB_LPT_OVERRIDES`` or replaces ``lpt_chunks`` clears
    them on entry and exit)."""
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
    for hidden in _arch_sizes(arch)
    for calibration in (
        ("record", "screen") if arch in OWN.get(_lpt(hidden), ()) else ("record",)
    )
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "hidden", "calibration"), _TWO_SIZE_CASES)
def test_two_size_rule(arch: str, sm: int, hidden: int, calibration: str) -> None:
    """At every token count of record and of ``LPT_SWEEP``, at every hidden size of the board's architecture, under
    the size's table of record (``BWD_SLAB_LPT`` at 4096, ``BWD_SLAB_LPT_HIDDEN`` at 5120 and 7168) and under the
    kernel test's screen ``(1.0, 0.99)`` where the architecture has the size's ``lpt`` record (with the size's unit
    order set to ``"chunk"`` at every token count and no board cell: no two-size record of the slab order ships):
    ``None`` never fires; ``bwd_slab_atomic_program`` is ``lpt`` exactly where ``lpt_chunks`` gives a geometry, else
    ``dynr``, with ``_os``
    where the board's unit-order band (:func:`_order_cell`: the size's ``BWD_SLAB_ORDER``, or the board's cell) is
    ``"slab"``, and the selection holds that one column-slab record of the size; ``slab_plan`` is the geometry, else
    the one-size chunks of ``host_plan``.  A geometry is 64-token multiples, ``1 <= p_big <= P <= 32``, big chunks
    within six tiles under the one-size chunks, covering the tokens: two sizes (the big chunks leave tokens, a smaller
    small chunk) or one (``p_big = P``, the last chunk holding tokens); its list-scheduling makespan over the size's
    ``H / 64`` slabs (a scheduler of this test) is at most ``keep`` times the one-size chunks'.  The firing token
    counts are printed; under the screen the rule fires on B200 (148 SMs) at every size."""
    table = _lpt_table(hidden)
    value = table[arch] if calibration == "record" else LPT_SCREEN
    fired, counted = [], 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(table, arch, value)
        if calibration == "screen":
            patch.setitem(nd.BWD_SLAB_ORDER[hidden], arch, ((None, "chunk"),))
            patch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {})
        _clear_plans()
        try:
            for T in sorted(set(T_VALUES) | set(LPT_SWEEP)):
                counted += 1
                what = (arch, sm, hidden, calibration, T)
                geometry = nd.lpt_chunks(T, sm, arch, hidden=hidden)
                if _calibration_at(value, T) is None:
                    assert geometry is None, what
                order = _band_at(T, _order_cell(arch, sm, hidden))
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
    """Under the tables of record (the board's cells where it has them) the two-size schedule runs at none of the
    benchmark token counts (``LPT_FIRES``): a table that starts firing would sum other token ranges per chunk than
    the deterministic column slab without a failing test.  The program is ``dynr`` / ``dynr_os``, ``_os`` exactly
    where the board's unit-order band (:func:`_order_cell`) is ``"slab"``: on sm_107a with 212 SMs ``dynr`` at 1024
    and 1536, ``dynr_os`` from 2048."""
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
        slab_order = _band_at(T, _order_cell(arch, sm)) == "slab"
        assert program.endswith("_os") == slab_order, (T, program)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "calibration", "fires"), LPT_FORMER_FIRES)
def test_two_size_rule_under_the_former_calibrations(
    arch: str, sm: int, calibration: tuple, fires: dict, monkeypatch
) -> None:
    """The rule is unchanged: with the board's former calibration written into ``BWD_SLAB_LPT`` (no board cells)
    ``lpt_chunks`` gives the geometries of ``LPT_FORMER_FIRES`` at their token counts, ``slab_plan`` binds them, and
    ``bwd_slab_atomic_program`` names ``lpt`` there (``lpt_os`` in a ``"slab"`` band of the table's unit order)."""
    monkeypatch.setitem(nd.BWD_SLAB_LPT, arch, calibration)
    monkeypatch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", {})
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {})
    _clear_plans()
    try:
        for T, geometry in fires.items():
            assert tuple(nd.lpt_chunks(T, sm, arch)) == geometry, T
            assert nd.slab_plan(T, sm, arch) == dict(
                zip(LPT_HOST_KEYS, geometry, strict=True)
            ), T
            slab_order = _band_at(T, nd.BWD_SLAB_ORDER[HIDDEN][arch]) == "slab"
            assert nd.bwd_slab_atomic_program(T, sm, arch) == "lpt" + (
                "_os" if slab_order else ""
            ), T
    finally:
        _clear_plans()


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "hidden", "fires"), LPT_FIRES_HIDDEN)
def test_two_size_rule_fires_where_calibrated_at_the_new_hidden_sizes(
    arch: str, sm: int, hidden: int, fires: dict
) -> None:
    """At hidden 5120 and 7168 under the table of record (``BWD_SLAB_LPT_HIDDEN``) the two-size schedule runs at none
    of the token counts of ``LPT_HIDDEN_T`` (``LPT_FIRES_HIDDEN``).  Under the kernel test's screen on B200, whose
    routes hold the ``lpt`` record of these sizes, it gives the geometries of ``LPT_SCREEN_FIRES_HIDDEN``, ``slab_plan``
    the geometry there, and the screen's plans are cached per hidden size: after the H 4096 plan of the same token
    count, SM count and ``o_tiles``, the size's plan is its own."""
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
        slab_order = _band_at(T, _order_cell(arch, sm, hidden)) == "slab"
        assert program.endswith("_os") == slab_order, (T, program)
    assert not fires, (arch, hidden)
    screened = [
        f for a, s, h, f in LPT_SCREEN_FIRES_HIDDEN if (a, s, h) == (arch, sm, hidden)
    ]
    assert bool(screened) == ((arch, sm) == ("sm_100a", 148)), (arch, sm)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(_lpt_table(hidden), arch, LPT_SCREEN)
        _clear_plans()
        try:
            for T, geometry in (screened[0] if screened else {}).items():
                assert nd.slab_plan(T, sm, arch, hidden) == dict(
                    zip(LPT_HOST_KEYS, geometry, strict=True)
                ), T
                at_4096 = nd._lpt_plan(T, sm, LPT_SCREEN[0])
                plan = nd._lpt_plan(T, sm, LPT_SCREEN[0], hidden=hidden)
                assert plan[2] == geometry and plan != at_4096, (T, plan, at_4096)
        finally:
            _clear_plans()


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
@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_forced_geometry_is_read_at_call_time(monkeypatch, hidden: int) -> None:
    """``lpt_chunks`` is a module global read at call time, at every hidden size (on B200, the size's order set to
    ``"chunk"`` for every token count): a replacement that forces a geometry at one token count moves ``slab_plan``
    and ``bwd_slab_atomic_program`` (``lpt``) there, and since no two-size record ships, the ``bwd_finish`` selection
    refuses the call (a ``ValueError``, never a backward without the column slab); the ``dynr`` record's launch grid
    (``min(SMs, H / 64 x P)`` over the one-size chunks) never moves.  A forced geometry that does not cover the tokens
    is a ``ValueError``; without the replacement the size's table decides (``None``: ``dynr`` and the one-size chunks,
    the selection's one column slab).  Each of these asks the rule at the size (the replacement records the hidden
    size it is asked at)."""
    arch, sm, T = "sm_100a", 148, 2049
    monkeypatch.setattr(nd, "sm_count", lambda _index: sm)
    monkeypatch.setattr(nd, "device_arch", lambda _index: arch)
    monkeypatch.setitem(nd.BWD_SLAB_ORDER[hidden], arch, ((None, "chunk"),))
    rule, forced, seen = nd.lpt_chunks, {}, []

    def forcing(t, s, a, hidden=HIDDEN):
        seen.append(hidden)
        return forced[t] if t in forced else rule(t, s, a, hidden=hidden)

    monkeypatch.setattr(nd, "lpt_chunks", forcing)
    slab = _record_of("dynr", hidden)
    assert _lpt(hidden) not in nd.MODULES, hidden
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
        assert nd.bwd_slab_atomic_program(T, sm, arch, hidden) == "lpt"
        with pytest.raises(ValueError, match="selects 0 column-slab records"):
            nd.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
        assert seen and set(seen) == {hidden}, seen
        assert nd.launch_grid(nd.MODULES[slab], T, device_index=0) == dynr_grid
        assert nd.slab_plan(T + 64, sm, arch, hidden) != dict(
            zip(LPT_HOST_KEYS, geometry, strict=True)
        )
    forced[T] = (2, 512, 1, 64)  # 576 tokens: not a partition of 2049
    with pytest.raises(ValueError, match="not a two-size geometry"):
        nd.slab_plan(T, sm, arch, hidden)
    del forced[T]
    assert _lpt_table(hidden)[arch] is None, hidden
    assert nd.bwd_slab_atomic_program(T, sm, arch, hidden) == "dynr"
    names = nd.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
    assert names[-1] == slab, names
    assert nd.slab_plan(T, sm, arch, hidden) == dict(
        zip(LPT_HOST_KEYS, one, strict=True)
    )
    assert nd.launch_grid(nd.MODULES[slab], T, device_index=0) == dynr_grid


@pytest.mark.cpu_only
def test_workspace_envelope_never_runs_the_screen(monkeypatch) -> None:
    """The workspace sizes and the rules rows never run the column slab's screen: ``host_plan``, which the envelope
    reads at every 64-token tile, carries no ``lpt`` value (its ``dfn`` slot count ``slot_parts`` is a function of
    the token count alone).  With ``lpt_chunks``, ``_lpt_plan`` and
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
    if arch not in OWN.get(_lpt(hidden), ())
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm", "hidden"), _ONE_SIZE_CASES)
def test_column_slab_rule_without_a_calibration_of_the_size(
    arch: str, sm: int, hidden: int, monkeypatch
) -> None:
    """At hidden 5120 and 7168 on an architecture without the size's ``lpt`` record the size's table is ``None`` and
    the column slab runs its one-size ``dynr`` record (``dynr_os`` where the board's unit-order band, the size's
    ``BWD_SLAB_ORDER`` or the board's cell: :func:`_order_cell`, is ``"slab"``) at every token count.  With
    ``_lpt_plan`` replaced by a function that raises (the rule never runs the screen there), at the token counts of
    record, of ``LPT_SWEEP`` and of ``LPT_HIDDEN_T``: ``lpt_chunks`` is ``None`` and ``bwd_slab_atomic_program``
    ``dynr`` / ``dynr_os``; ``slab_plan`` is the one-size chunks of ``k7_chunks`` at the size (``P`` and the rows per
    chunk of the deterministic ``host_plan`` of the size: both backwards sum the partials of the same chunks); where
    the architecture routes the size the ``bwd_finish`` selection holds that record of the size.  The order of some
    of these token counts is the board's cell's, not the table's, on GB300 at 7168 (:data:`GB300_CELL`) alone."""

    def screen(*_args, **_kwargs):
        raise AssertionError(
            f"the column-slab rule of hidden {hidden} on {arch} ran the lpt screen"
        )

    assert _lpt_table(hidden)[arch] is None, (arch, hidden)
    monkeypatch.setattr(nd, "_lpt_plan", screen)
    checked = by_cell = 0
    for T in sorted(set(T_VALUES) | set(LPT_SWEEP) | set(LPT_HIDDEN_T)):
        what = (arch, sm, hidden, T)
        assert nd.lpt_chunks(T, sm, arch, hidden=hidden) is None, what
        order = _band_at(T, _order_cell(arch, sm, hidden))
        by_cell += order != _band_at(T, nd.BWD_SLAB_ORDER[hidden][arch])
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
    assert (by_cell > 0) == ((arch, sm, hidden) == GB300_CELL), by_cell


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


# A banded calibration the rule is checked under (bands are a form of BWD_SLAB_LPT, whose entries of record are not
# banded; the board cell of record of BWD_SLAB_LPT_OVERRIDES is) and the token counts it is checked at: both sides of
# each band edge, below and above them.
_BANDS = ((33600, (0.5, 0.985)), (34368, (0.5, 0.98)), (None, (0.5, 0.97)))
_BAND_T = (1, 2049, 16263, 32768, 33599, 33600, 33601, 34368, 34369, 40000, 65536)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm", (208, 212, 216))
def test_banded_lpt_calibration(sm: int) -> None:
    """``BWD_SLAB_LPT`` is the table of record (``LPT_OF_RECORD``); every entry of it and of ``BWD_SLAB_LPT_HIDDEN``
    is ``None``, one ``(o_tiles, keep)`` with ``o_tiles > 0`` and ``0 < keep <= 1``, or token-count bands of such
    values ending in an unbounded one.  With the board cells emptied (``BWD_SLAB_ORDER_OVERRIDES``,
    ``BWD_SLAB_LPT_OVERRIDES``: the 212-SM board reads the table too) and sm_107a's entry banded (``_BANDS``) at 208,
    212 and 216 SMs the calibration of a token count is ``(0.5, 0.985)`` to 33600 tokens, ``(0.5, 0.98)`` to 34368 and
    ``(0.5, 0.97)`` above (both sides of each edge), and ``lpt_chunks`` is the rule under the scalar table of that
    calibration; bands that leave a token count uncovered are a ``ValueError``."""
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
            patch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {})
            patch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", {})
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
    where the architecture has that record of the size (on sm_107a at 208 and 216 SMs too: its H 4096 route holds
    the ``_os`` records for the 212-SM board's cell; where an architecture has none the ``"slab"`` band is a
    ``ValueError`` of the selection, never a backward without the column slab) and 2050 tokens the form itself.  The
    table decides on the boards of record but GB300 at 7168, whose order cell (:data:`GB300_CELL`, the one board cell
    of a board of record) decides there instead: 2049 and 2050 tokens select the form of the cell's first band, the
    chunk order, whatever the table names."""
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
    cells = {h for a, s, h in nd.BWD_SLAB_ORDER_OVERRIDES if (a, s) == (arch, sm)}
    assert cells == ({GB300_CELL[2]} if (arch, sm) == GB300_CELL[:2] else set()), cells
    with pytest.MonkeyPatch.context() as patch:
        _clear_plans()
        try:
            for hidden in _arch_sizes(arch):
                bands = ((2049, "slab"), (None, "chunk"))
                patch.setitem(nd.BWD_SLAB_ORDER[hidden], arch, bands)
                for T, suffix in ((2049, "_os"), (2050, "")):
                    what = (hidden, T)
                    # the board's cell decides there, not the patched table
                    if hidden in cells:
                        cell = ORDER_OVERRIDES[arch, sm, hidden]
                        assert _band_at(T, cell) == "chunk", (what, cell[0])
                        suffix = ""
                    geometry = nd.lpt_chunks(T, sm, arch, hidden=hidden)
                    program = nd.bwd_slab_atomic_program(T, sm, arch, hidden)
                    assert program == ("lpt" if geometry else "dynr") + suffix, what
                    if arch not in OWN.get(_record_of(program, hidden), ()):
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


def _valid_lpt_cell(cell) -> bool:
    """A calibration entry: ``None``, one ``(o_tiles, keep)`` or token-count bands of such values."""
    if cell is not None and isinstance(cell[0], tuple):
        return _valid_bands(cell, _valid_calibration)
    return _valid_calibration(cell)


@pytest.mark.cpu_only
def test_slab_overrides_of_record() -> None:
    """``BWD_SLAB_ORDER_OVERRIDES`` / ``BWD_SLAB_LPT_OVERRIDES`` are the board cells of record (:data:`ORDER_OVERRIDES`,
    :data:`LPT_OVERRIDES`), exported and well formed: each key is ``(arch, sm, hidden)`` with a known architecture, an
    SM count that is a positive int (never ``None``: the table itself is the cell of every SM count) and a hidden size
    the architecture routes (an int, never ``None``: a cell of every size would apply a 4096 calibration at 5120 and
    7168); an order cell is token-count bands of ``"chunk"`` / ``"slab"``, a calibration cell ``None``, one
    ``(o_tiles, keep)`` or bands of such values (no calibration cell is of record).  On each board with a cell, on
    both sides of every band edge of its
    two cells: every program they can name has its record on the ``bwd_finish`` route of the size (a ``"slab"`` band
    the ``dynr_os`` record, and ``lpt_os`` where the calibration is not ``None``); ``bwd_slab_order`` and
    ``_lpt_calibration`` read the cells with the board's SM count and the tables without one.  Below the order cell's
    first ``"slab"`` band the calibration cell has no band of its own (its first bound at or above that band's lower
    edge) and reads the table's calibration: the bands start where the slab order does."""
    assert nd.BWD_SLAB_ORDER_OVERRIDES == ORDER_OVERRIDES
    assert nd.BWD_SLAB_LPT_OVERRIDES == LPT_OVERRIDES
    assert {"BWD_SLAB_ORDER_OVERRIDES", "BWD_SLAB_LPT_OVERRIDES"} <= set(nd.__all__)
    for cells, valid in (
        (
            nd.BWD_SLAB_ORDER_OVERRIDES,
            lambda cell: _valid_bands(cell, lambda order: order in ("chunk", "slab")),
        ),
        (nd.BWD_SLAB_LPT_OVERRIDES, _valid_lpt_cell),
    ):
        assert cells or cells is nd.BWD_SLAB_LPT_OVERRIDES
        for key, cell in cells.items():
            arch, sm, hidden = key
            assert arch in nd.ARCHES, key
            assert type(sm) is int and sm > 0, key
            assert type(hidden) is int, key
            assert hidden in nd.supported_hidden_sizes(arch), key
            assert valid(cell), (key, cell)
    boards = sorted({*nd.BWD_SLAB_ORDER_OVERRIDES, *nd.BWD_SLAB_LPT_OVERRIDES})
    for arch, sm, hidden in boards:
        key = (arch, sm, hidden)
        order, entry = _order_cell(arch, sm, hidden), _lpt_cell(arch, sm, hidden)
        table_order = nd.BWD_SLAB_ORDER[hidden][arch]
        table_entry = _lpt_table(hidden)[arch]
        bands = entry if entry is not None and isinstance(entry[0], tuple) else ()
        edges = sorted({t for t, _ in (*order, *bands) if t is not None})
        route = nd.route_modules(arch, hidden, "bwd_finish")
        for T in sorted({1, *edges, *(t + 1 for t in edges)}):
            what = (key, T)
            calibration = _calibration_at(entry, T)
            assert nd.bwd_slab_order(T, arch, hidden, sm_count=sm) == _band_at(
                T, order
            ), what
            assert nd._lpt_calibration(T, arch, hidden, sm_count=sm) == calibration, (
                what
            )
            assert nd.bwd_slab_order(T, arch, hidden) == _band_at(T, table_order), what
            assert nd._lpt_calibration(T, arch, hidden) == _calibration_at(
                table_entry, T
            ), what
            suffix = "_os" if _band_at(T, order) == "slab" else ""
            programs = ["dynr" + suffix] + (
                ["lpt" + suffix] if calibration is not None else []
            )
            for program in programs:
                assert _record_of(program, hidden) in route, (what, program)
        first_slab = [i for i, (_t, o) in enumerate(order) if o == "slab"]
        if first_slab and first_slab[0] > 0:
            chunk_to = order[first_slab[0] - 1][0]
            if bands:
                assert bands[0][0] is None or bands[0][0] >= chunk_to, (key, bands)
            for T in (1, chunk_to):
                assert _calibration_at(entry, T) == _calibration_at(table_entry, T), (
                    key,
                    T,
                )


@pytest.mark.cpu_only
def test_gb300_cell_follows_the_chunk_count() -> None:
    """The shipped GB300 order cell (``BWD_SLAB_ORDER_OVERRIDES[GB300_CELL]``) holds where it was measured: its
    ``"slab"`` bands below 63092 are the token-count segments of :data:`GB300_CELL_SPAN` whose column-slab chunk count
    ``P`` (``k7_chunks`` at 152 SMs and hidden 7168: the chunks ``slab_plan`` and the launch grid read) is at least 19
    and where the slab order read faster (all of them but :data:`GB300_CELL_TIE`).  Over every token count of the span:
    each one in a ``"slab"`` band has ``P >= 19`` (18355 of them), so each one with ``P < 19`` lies in a ``"chunk"``
    band, and each one in a ``"chunk"`` band has ``P < 19`` but the segment that keeps the chunk order
    (:data:`GB300_CELL_TIE`, ``P`` 19) -- a change of ``k7_chunks`` or of the cell that moves the chunk count under the
    bands fails here, not on a device.  Outside the span the cell names the table's order: ``"chunk"`` to 33792 (its
    first band; ``P`` 4 at 33792, 19 at 33793 and at 63091) and ``"slab"`` from 63092 (its last band, which starts below
    the span's end)."""
    arch, sm, hidden = GB300_CELL
    cell = nd.BWD_SLAB_ORDER_OVERRIDES[GB300_CELL]
    table = nd.BWD_SLAB_ORDER[hidden][arch]
    (lo, hi), (tie_lo, tie_hi) = GB300_CELL_SPAN, GB300_CELL_TIE
    assert table == ((hi, "chunk"), (None, "slab")), table
    assert cell[0] == (lo - 1, "chunk") and cell[-1] == (None, "slab"), cell
    assert cell[-2][0] < hi, cell
    assert _band_at(tie_lo, cell) == _band_at(tie_hi, cell) == "chunk", cell
    assert [nd.k7_chunks(T, sm, hidden=hidden)[0] for T in (lo - 1, lo, hi)] == [
        4,
        19,
        19,
    ]
    slab_t, previous = 0, 0
    for t_upper, order in cell:
        upper = hi if t_upper is None else min(t_upper, hi)
        for T in range(max(previous + 1, lo), upper + 1):
            P = nd.k7_chunks(T, sm, hidden=hidden)[0]
            if order == "slab":
                assert P >= GB300_CELL_MIN_P, (T, P, t_upper)
                slab_t += 1
            elif tie_lo <= T <= tie_hi:
                assert P == GB300_CELL_MIN_P, (T, P)
            else:
                assert P < GB300_CELL_MIN_P, (T, P, t_upper)
        if t_upper is None or t_upper >= hi:
            break
        previous = t_upper
    assert slab_t == GB300_CELL_SLAB_T, slab_t


# A test table for the lookup of the board cells, which reads the cells of the exact key (arch, SM count, hidden
# size) alone: the cells of sm_107a with 212 SMs at 4096 differ from the tables at every token count of
# SLAB_PRECEDENCE_T, and the keys with None (no SM count; every hidden size) match no call.
SLAB_PRECEDENCE_ORDER = {
    ("sm_107a", 212, 4096): ((2048, "slab"), (None, "chunk")),
    ("sm_107a", None, 4096): ((None, "slab"),),
    ("sm_107a", 216, None): ((None, "slab"),),
    ("sm_100a", 212, 4096): ((None, "chunk"),),
}
SLAB_PRECEDENCE_LPT = {
    ("sm_107a", 212, 4096): ((2048, None), (None, (1.0, 0.99))),
    ("sm_107a", None, 4096): (1.0, 0.95),
    ("sm_107a", 216, None): (1.0, 0.95),
    ("sm_100a", 212, 4096): None,
}
SLAB_PRECEDENCE_T = (1, 2048, 2049, 65536)
# (arch, SM count, the key whose cells the calls read: None for the tables BWD_SLAB_ORDER / BWD_SLAB_LPT).
SLAB_PRECEDENCE = (
    ("sm_107a", 212, ("sm_107a", 212, 4096)),  # (arch, sm, hidden)
    ("sm_107a", 216, None),  # the tables, not (arch, 216, None): a key names its hidden size
    ("sm_107a", 208, None),  # the tables: no key names the board
    ("sm_107a", None, None),  # the tables, not (arch, None, 4096): a call without an SM count
    ("sm_100a", 212, ("sm_100a", 212, 4096)),  # (arch, sm, hidden) of another architecture
    ("sm_100a", 148, None),  # the tables
)  # fmt: skip


@pytest.mark.cpu_only
def test_slab_override_lookup(monkeypatch) -> None:
    """The lookup of the board cells under a test table (:data:`SLAB_PRECEDENCE_ORDER`, :data:`SLAB_PRECEDENCE_LPT`):
    a call with an SM count reads the cells of the exact key ``(arch, sm_count, hidden)``; every other call -- another
    SM count or architecture, no SM count -- reads ``BWD_SLAB_ORDER`` / ``BWD_SLAB_LPT``, and the keys with ``None``
    match nothing (:data:`SLAB_PRECEDENCE`).  ``bwd_slab_atomic_program`` and ``lpt_chunks`` pass the SM count:
    ``_os`` exactly where the order read is ``"slab"``, no geometry where the calibration read is ``None``.  An error
    names what it read (the cell's key, or the table, architecture and size), and ``sm_count`` is a keyword: as a
    fourth positional argument it is a ``TypeError``, never a value bound to another parameter."""
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", SLAB_PRECEDENCE_ORDER)
    monkeypatch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", SLAB_PRECEDENCE_LPT)
    on_212 = []
    for arch, sm, key in SLAB_PRECEDENCE:
        for T in SLAB_PRECEDENCE_T:
            what = (arch, sm, T)
            if key is None:
                order = _band_at(T, ORDER[HIDDEN][arch])
                calibration = _calibration_at(LPT_OF_RECORD[arch], T)
            else:
                order = _band_at(T, SLAB_PRECEDENCE_ORDER[key])
                calibration = _calibration_at(SLAB_PRECEDENCE_LPT[key], T)
            assert nd.bwd_slab_order(T, arch, HIDDEN, sm_count=sm) == order, what
            assert nd._lpt_calibration(T, arch, HIDDEN, sm_count=sm) == calibration, (
                what
            )
            if (arch, sm) == ("sm_107a", 212):
                on_212.append((order, calibration))
            if sm is None:
                continue
            geometry = nd.lpt_chunks(T, sm, arch)
            if calibration is None:
                assert geometry is None, what
            program = nd.bwd_slab_atomic_program(T, sm, arch)
            assert program == ("lpt" if geometry else "dynr") + (
                "_os" if order == "slab" else ""
            ), what
    assert on_212 == [
        ("slab", None),
        ("slab", None),
        ("chunk", (1.0, 0.99)),
        ("chunk", (1.0, 0.99)),
    ]
    key = ("sm_107a", 212, 4096)
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {key: ((2048, "slab"),)})
    monkeypatch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", {key: ((2048, (1.0, 0.99)),)})
    with pytest.raises(ValueError) as error:
        nd.bwd_slab_order(2049, "sm_107a", sm_count=212)
    assert str(error.value).startswith(
        f"BWD_SLAB_ORDER_OVERRIDES[{key!r}] leaves T = 2049 uncovered"
    ), error.value
    with pytest.raises(ValueError) as error:
        nd.lpt_chunks(2049, 212, "sm_107a")
    assert str(error.value).startswith(
        f"BWD_SLAB_LPT_OVERRIDES[{key!r}] leaves T = 2049 uncovered"
    ), error.value
    assert nd.bwd_slab_order(2049, "sm_107a", sm_count=216) == "chunk"
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {key: ((None, "slabs"),)})
    with pytest.raises(ValueError) as error:
        nd.bwd_slab_order(1, "sm_107a", sm_count=212)
    assert str(error.value).startswith(
        f"BWD_SLAB_ORDER_OVERRIDES[{key!r}] names the unit order 'slabs'"
    ), error.value
    monkeypatch.setitem(nd.BWD_SLAB_ORDER[HIDDEN], "sm_107a", ((2048, "chunk"),))
    with pytest.raises(ValueError) as error:
        nd.bwd_slab_order(2049, "sm_107a", sm_count=216)
    assert str(error.value).startswith(
        "BWD_SLAB_ORDER of sm_107a at hidden 4096 leaves T = 2049 uncovered"
    ), error.value
    with pytest.raises(TypeError):
        nd.bwd_slab_order(2049, "sm_107a", HIDDEN, 212)
    with pytest.raises(TypeError):
        nd._lpt_calibration(2049, "sm_107a", HIDDEN, 212)


def _cell_changes(arch: str, sm: int, hidden: int, T: int) -> bool:
    """Whether the board cells of record change the column slab of the board ``(arch, sm)`` at ``hidden`` and ``T``:
    where the board's order cell names another order than the size's ``BWD_SLAB_ORDER`` -- on R200 with 212 SMs at
    4096 from 1877 tokens (its calibration cell differs from the table there alone), on GB300 at 7168 in the cell's
    ``"slab"`` bands in :data:`GB300_CELL_SPAN`."""
    key = (arch, sm, hidden)
    return key in ORDER_OVERRIDES and _band_at(T, ORDER_OVERRIDES[key]) != _band_at(
        T, ORDER[hidden][arch]
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), sorted({*SLAB_BOARDS, *PDL_BOARDS}))
def test_slab_overrides_change_only_their_board(
    arch: str, sm: int, monkeypatch
) -> None:
    """The board cells of record change the column slab of their own board alone: on every board (the boards of
    record, GB200, B300 and R200 with 212 SMs), at every hidden size the architecture routes, at the token counts of
    record and of ``LPT_SWEEP`` and on both sides of every band edge of the cells and of the table band the GB300 cell
    runs into (:data:`OVERRIDE_EDGES`: 33792 / 33793 and 63091 / 63092 among them), ``bwd_slab_atomic_program``,
    ``slab_plan``, the ``bwd_finish`` selection and its column-slab launch's per-call PDL argument on every path equal
    their values with both tables of board cells emptied -- except on sm_107a with 212 SMs at hidden 4096 from 1877
    tokens and on sm_103a with 152 SMs at hidden 7168 in the ``"slab"`` bands of its cell up to 63091
    (:func:`_cell_changes`).  There the program is the ``_os`` copy of a form where the tables name a chunk-order
    form, the selection holds its record as its one column slab after the same token owner and fn split, and the PDL
    argument is unchanged.  On every board each value is also its value with the board's cells (where it has them)
    written into ``BWD_SLAB_ORDER`` / ``BWD_SLAB_LPT`` instead: a cell is the table entry of its board."""
    token_counts = sorted({*T_VALUES, *LPT_SWEEP, *OVERRIDE_EDGES})

    def column_slab() -> dict:
        found = {}
        for hidden in _arch_sizes(arch):
            for T in token_counts:
                names = nd.selected_modules(
                    arch, hidden, "bwd_finish", rows=T, sm_count=sm
                )
                slabs = [n for n in names if nd.MODULES[n]["role"] == "bwd_slab"]
                assert len(slabs) == 1, (hidden, T, names)
                found[hidden, T] = (
                    nd.bwd_slab_atomic_program(T, sm, arch, hidden),
                    nd.slab_plan(T, sm, arch, hidden),
                    names,
                    tuple(
                        nd.enable_pdl(slabs[0], T, arch, path, sm_count=sm)
                        for path in nd.PDL_PATHS
                    ),
                )
        return found

    shipped = column_slab()
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {})
    monkeypatch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", {})
    tables = column_slab()
    own = [h for h in _arch_sizes(arch) if (arch, sm, h) in ORDER_OVERRIDES]
    for hidden in own:
        key = (arch, sm, hidden)
        monkeypatch.setitem(nd.BWD_SLAB_ORDER[hidden], arch, ORDER_OVERRIDES[key])
        if key in LPT_OVERRIDES:
            monkeypatch.setitem(_lpt_table(hidden), arch, LPT_OVERRIDES[key])
    cells = column_slab() if own else tables
    changed = set()
    for (hidden, T), (program, _plan, names, pdl) in shipped.items():
        what = (hidden, T)
        assert shipped[hidden, T] == cells[hidden, T], what
        if not _cell_changes(arch, sm, hidden, T):
            assert shipped[hidden, T] == tables[hidden, T], what
            continue
        table_program, _table_plan, table_names, table_pdl = tables[hidden, T]
        assert program.endswith("_os") and not table_program.endswith("_os"), (
            what,
            program,
            table_program,
        )
        assert names[:-1] == table_names[:-1], (what, names, table_names)
        assert names[-1] == _record_of(program, hidden), (what, names)
        assert pdl == table_pdl, what
        changed.add((hidden, T))
    assert bool(changed) == bool(own), (own, changed)
    if (arch, sm) == ("sm_107a", 212):
        assert changed == {(HIDDEN, T) for T in token_counts if T > 1876}, changed
    if (arch, sm) == GB300_CELL[:2]:
        (lo, hi), cell = GB300_CELL_SPAN, ORDER_OVERRIDES[GB300_CELL]
        assert changed == {
            (GB300_CELL[2], T)
            for T in token_counts
            if lo <= T <= hi and _band_at(T, cell) == "slab"
        }, changed


def _route_plan_launches(
    monkeypatch, arch: str, hidden: int, sm: int, token_counts: tuple
) -> dict:
    """The backward's launches in the route plans of a device of ``arch`` with ``sm`` SMs at ``hidden``, per token
    count of ``token_counts``: ``[(record, {logical name: bound scalar})]`` in launch order (the device's SM count,
    architecture and module loader replaced: nothing is built or launched; both plan caches emptied first, since
    they are keyed by the device index, not its SM count)."""
    path = "backward"
    values = (
        *pkg._CALL_VALUES["bwd_finish"],
        *(name for name, _ in pkg._WORKSPACE_VALUES["bwd_finish"]),
    )
    placeholders = tuple(f"<{name}>" for name in values)

    class _Module:
        def __init__(self, name: str) -> None:
            self.name = name

        def __getattr__(self, entry: str) -> tuple[str, str]:
            return (self.name, entry)

    monkeypatch.setattr(nd, "device_arch", lambda _index: arch)
    monkeypatch.setattr(nd, "load", lambda name, _arch: _Module(name))
    monkeypatch.setattr(nd, "sm_count", lambda _index: sm)
    monkeypatch.setattr(pkg, "_PLANS", {})
    monkeypatch.setattr(pkg, "_FAST_ROUTES", {})
    found = {}
    for T in token_counts:
        route = pkg._fast_route(arch, "bwd_finish", T, hidden, 0, path)
        launches = []
        for (name, _entry), arguments in route.launches:
            record = nd.MODULES[name]
            bound = {
                record["logical"][key]: value
                for (kind, key), value in zip(
                    record["arg_plan"],
                    arguments(placeholders + route.statics),
                    strict=True,
                )
                if kind == "parameter"
            }
            launches.append((name, bound))
        found[T] = launches
    return found


def _route_plan_slabs(
    monkeypatch, arch: str, hidden: int, sm: int, token_counts: tuple
) -> dict:
    """The backward's column-slab launch in the route plans of :func:`_route_plan_launches`, per token count of
    ``token_counts``: ``(record, lpt)``.  Each plan holds one column-slab launch, the record
    ``bwd_slab_atomic_program`` names with the device's SM count; ``lpt`` is the ``lpt_*`` values an ``lpt`` /
    ``lpt_os`` launch binds, those of ``slab_plan`` (``None`` for another record, which binds none of them)."""
    found = {}
    for T, launches in _route_plan_launches(
        monkeypatch, arch, hidden, sm, token_counts
    ).items():
        what = (sm, T)
        slabs = [
            (name, bound)
            for name, bound in launches
            if nd.MODULES[name]["role"] == "bwd_slab"
        ]
        assert len(slabs) == 1, (what, [name for name, _ in launches])
        ((name, bound),) = slabs
        program = nd.bwd_slab_atomic_program(T, sm, arch, hidden)
        assert name == _record_of(program, hidden), (what, name, program)
        lpt = None
        if name in (_lpt(hidden), _lpt_os(hidden)):
            lpt = tuple(bound[key] for key in LPT_HOST_KEYS)
            plan = nd.slab_plan(T, sm, arch, hidden)
            assert lpt == tuple(plan[key] for key in LPT_HOST_KEYS), (what, lpt)
        else:
            assert not set(LPT_HOST_KEYS) & set(bound), (what, bound)
        found[T] = (name, lpt)
    return found


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), det.RULES_BOARDS)
def test_route_plan_binds_the_parameter_rows(monkeypatch, arch: str, sm: int) -> None:
    """The route plans (the count of record) bind the parameter rows the column slab folds to the rows the token
    owner of the plan writes: at every hidden size of the board's architecture, on both sides of every
    ``BWD_ROW_BANDS`` edge of the size and at 1, 2049 and 65536 tokens, the plan launches the token owner of the
    selection -- its group-row form ``_trows_dyn`` exactly where ``bwd_group_rows`` holds -- and the column slab
    binds ``prow`` 1 and ``param_rows`` = the group count ``ceil(T / 8)`` there, ``prow`` 0 and ``param_rows`` = the
    token owner's CTA count ``G6`` elsewhere (:func:`_fold_plan`)."""
    checked = set()
    for hidden in _arch_sizes(arch):
        edges = {
            t + side
            for above, at_most in nd._tab("BWD_ROW_BANDS", hidden)[arch]
            for t in (above, at_most)
            if t is not None
            for side in (0, 1)
        }
        token_counts = tuple(sorted(edges | {1, 2049, 65536}))
        launches = _route_plan_launches(monkeypatch, arch, hidden, sm, token_counts)
        for T, plan in launches.items():
            what = (hidden, T)
            group_rows = nd.bwd_group_rows(T, sm, arch, hidden)
            selection = nd.selected_modules(
                arch, hidden, "bwd_finish", rows=T, sm_count=sm
            )
            assert [name for name, _ in plan] == list(selection), (what, plan)
            (token,) = [n for n in selection if nd.MODULES[n]["role"] == "bwd_token"]
            stem = token.removesuffix(f"__h{hidden}")
            assert stem.endswith(GROUP_ROWS) == group_rows, (what, token)
            (bound,) = [b for n, b in plan if nd.MODULES[n]["role"] == "bwd_slab"]
            want = _fold_plan(det.host_plan(T, sm, arch, hidden))
            assert bound["prow"] == want["prow"] == int(group_rows), (what, bound)
            assert bound["param_rows"] == want["param_rows"], (what, bound)
            assert bound["param_rows"] == (
                -(-T // det.TOKEN_GROUP)
                if group_rows
                else det.k6_grid(T, sm, hidden)[0]
            ), (what, bound)
            checked.add(group_rows)
    assert checked == ({False, True} if arch in NEW_HIDDEN_ARCHES else {False}), checked


@pytest.mark.cpu_only
def test_route_plan_binds_the_board_cells(monkeypatch) -> None:
    """The route plans bind the column slab of the device's board: on sm_107a with 208, 212 and 216 SMs (the device's
    SM count, architecture and module loader replaced: nothing is built or launched; both plan caches emptied for
    every board and table state, since they are keyed by the device index, not its SM count), at 1876, 1877, 4096 and
    15566 tokens, the backward's one column-slab launch is the record ``bwd_slab_atomic_program`` names with the
    device's SM count, and no launch binds the ``lpt_*`` values (no table of record picks the two-size schedule).
    Against the plans with both tables of board cells emptied, only the 212-SM board's differ, and only from 1877
    tokens: ``dynr_os`` at 1877, 4096 and 15566 where the tables run ``dynr``."""
    arch, hidden = "sm_107a", HIDDEN
    boards, token_counts = (208, 212, 216), (1876, 1877, 4096, 15566)
    shipped = {
        sm: _route_plan_slabs(monkeypatch, arch, hidden, sm, token_counts)
        for sm in boards
    }
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {})
    monkeypatch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", {})
    tables = {
        sm: _route_plan_slabs(monkeypatch, arch, hidden, sm, token_counts)
        for sm in boards
    }
    for sm in boards:
        if sm != 212:
            assert shipped[sm] == tables[sm], (sm, shipped[sm], tables[sm])
    assert shipped[212][1876] == tables[212][1876], shipped[212][1876]
    assert {T: tables[212][T] for T in token_counts[1:]} == {
        1877: (SLAB, None),
        4096: (SLAB, None),
        15566: (SLAB, None),
    }, tables[212]
    assert {T: shipped[212][T] for T in token_counts[1:]} == {
        1877: (SLAB_OS, None),
        4096: (SLAB_OS, None),
        15566: (SLAB_OS, None),
    }, shipped[212]


@pytest.mark.cpu_only
def test_route_plan_binds_the_gb300_cell(monkeypatch) -> None:
    """The route plans bind the GB300 order cell (:data:`GB300_CELL`) on its board alone: on sm_103a at hidden 7168
    with 148 SMs (B300, no cell) and 152 SMs (GB300; :func:`_route_plan_slabs`), at 33920 and 37952 tokens (in
    ``"slab"`` bands of the cell: chunk counts 19 and 27 at 152 SMs), 49152 (in a ``"chunk"`` band: 4) and 63092 (the
    tables' ``"slab"`` band), the backward's one column-slab launch is ``dynr__h7168`` or ``dynr_os__h7168`` (sm_103a
    routes no lpt record of the size).  Against the plans with both tables of board cells emptied only the 152-SM
    board's differ, at 33920 and 37952: ``dynr_os__h7168`` where the tables run ``dynr__h7168``; at 49152 both boards
    run ``dynr__h7168`` and at 63092 ``dynr_os__h7168``."""
    arch, sm, hidden = GB300_CELL
    boards, token_counts = (148, sm), (33920, 37952, 49152, 63092)
    assert [nd.k7_chunks(T, sm, hidden=hidden)[0] for T in token_counts[:3]] == [
        19,
        27,
        4,
    ]
    shipped = {
        board: _route_plan_slabs(monkeypatch, arch, hidden, board, token_counts)
        for board in boards
    }
    monkeypatch.setattr(nd, "BWD_SLAB_ORDER_OVERRIDES", {})
    monkeypatch.setattr(nd, "BWD_SLAB_LPT_OVERRIDES", {})
    tables = {
        board: _route_plan_slabs(monkeypatch, arch, hidden, board, token_counts)
        for board in boards
    }
    dynr, dynr_os = (_slab(hidden), None), (_slab_os(hidden), None)
    assert shipped[148] == tables[148] == tables[sm], (shipped[148], tables)
    assert tables[sm] == {33920: dynr, 37952: dynr, 49152: dynr, 63092: dynr_os}, (
        tables[sm]
    )
    assert shipped[sm] == {
        33920: dynr_os,
        37952: dynr_os,
        49152: dynr,
        63092: dynr_os,
    }, shipped[sm]


@pytest.mark.cpu_only
def test_exported_hidden_sizes_per_architecture() -> None:
    """The (architecture, hidden size) pairs this package exports, on which the tests of this module compare it with
    the deterministic records and routes: 4096, 5120 and 7168 on sm_100a / sm_103a, 4096 alone on sm_107a.  The
    deterministic package exports each of them (and 5120 and 7168 on sm_107a too, which have no copy here)."""
    assert ND_HIDDEN_SIZES == {
        "sm_100a": (4096, 5120, 7168),
        "sm_103a": (4096, 5120, 7168),
        "sm_107a": (4096,),
    }
    for arch in det.ARCHES:
        assert tuple(nd.supported_hidden_sizes(arch)) == ND_HIDDEN_SIZES[arch], arch
        assert set(ND_HIDDEN_SIZES[arch]) <= set(det.supported_hidden_sizes(arch)), arch


@pytest.mark.cpu_only
def test_route_glue_and_supported_hidden_sizes() -> None:
    """The hidden sizes of each architecture are this package's (``ND_HIDDEN_SIZES``): 4096, 5120 and 7168 on
    sm_100a / sm_103a, 4096 on sm_107a, none on another architecture; a route applies exactly where the
    deterministic one does at those sizes (the deterministic package also routes 5120 and 7168 on sm_107a); a size
    or architecture without a route is a ``ValueError`` of the selection, and a size the package does
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
                det_applies = det.route_applies(
                    device_capability=capability, hidden=hidden, kind=kind
                )
                assert applies == (det_applies and hidden in _arch_sizes(arch)), (
                    capability,
                    hidden,
                    kind,
                )
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
    assert pkg.cake_mhc_nondet_train_route_modules is nd.route_modules
    assert pkg.cake_mhc_nondet_train_route_key is nd.route_key
    assert not hasattr(pkg, "route_key") and not hasattr(pkg, "route_modules")


@pytest.mark.cpu_only
def test_library_names_and_record_scalars() -> None:
    """``library_name`` is ``<library>_<arch>`` on the record's architectures and a ``RuntimeError`` elsewhere;
    ``RECORD_SCALARS`` is the deterministic table of every hidden size renamed without the backward fn splits and the
    checkpoint carry's records, ``CARRY_ROLES`` (the forward fn split of each size ``want_t = 0``; the atomic fn
    splits take no ``want_t``)."""
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
        if _det_role(name) not in ("fn_split_bwd", *CARRY_ROLES)
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
    "supported_cpu": package.is_cake_mhc_nondet_train_supported(torch.device("cpu")),
    "selection": {
        f"{arch}/{sm}/{hidden}": list(jit.selected_modules(arch, hidden, "bwd_finish", rows=4096, sm_count=sm))
        for arch, sm in jit.RULES_BOARDS
        for hidden in jit.supported_hidden_sizes(arch)
    },
    "bwd_bytes": jit.workspace_bytes("bwd", 4096, 148, "sm_100a"),
}
try:
    package.cake_mhc_nondet_train_backward_finish(*([None] * 9), deterministic=False)
except TypeError as error:
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
    sources) imports, plans its routes (every hidden size of every board) and refuses a ``deterministic`` keyword
    (the interpreter's ``TypeError``) in an interpreter where every ``flashinfer`` import raises; no flashinfer module
    is loaded."""
    neutral = "vendored_cake_mhc_nondet_train"
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
    assert _NO_MODE_KEYWORD in out["refusal"], out["refusal"]


# ----------------------------------------------------------------------------------------------------------------
# The mode policy (no device: a keyword the entry points do not take is refused before any operand, device or route
# check; the top-level module is read as source)


@pytest.mark.cpu_only
def test_exports() -> None:
    """Every exported name exists and, but the ALL_CAPS constants, carries the mode token ``nondet`` (``Nondet`` in a
    class name) and no ``det`` token; no head entry point; the former generic route helpers are gone."""
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
        assert "head" not in name.lower(), name
        if not name.isupper():
            assert _mode_tokens(name) == {"nondet"}, name
    assert set(ENTRIES) | set(HELPERS) <= set(pkg.__all__)
    assert {"CakeMHCNondetPrepareFunction", "CakeMHCNondetFinishFunction"} <= set(
        pkg.__all__
    )
    assert not {"route_key", "route_modules"} & set(pkg.__all__)
    assert pkg.SUPPORTED_COMPUTE_CAPABILITIES == ((10, 0), (10, 3), (10, 7))
    assert pkg.__name__ == PACKAGE_NAME


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    """The module-level functions of ``path`` by name."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _body(function: ast.FunctionDef) -> list[ast.stmt]:
    """The statements of ``function`` after its docstring."""
    first = function.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return function.body[1:]
    return function.body


def _expression(source: str) -> str:
    """``ast.dump`` of the expression ``source`` (no positions), to compare a statement's expression with."""
    return ast.dump(ast.parse(source, mode="eval").body)


@pytest.mark.cpu_only
def test_no_public_callable_takes_a_deterministic_keyword() -> None:
    """No public callable of the package has a ``deterministic`` parameter (the mode is the package: its names carry
    ``nondet``), the ``forward`` / ``backward`` of its autograd ``Function`` classes included, and the former mode
    check ``_require_nondeterministic`` is gone."""
    checked = []
    for name in pkg.__all__:
        obj = getattr(pkg, name)
        if inspect.isclass(obj):
            functions = {
                f"{name}.forward": obj.forward,
                f"{name}.backward": obj.backward,
            }
        elif callable(obj):
            functions = {name: obj}
        else:
            continue
        for label, function in functions.items():
            assert "deterministic" not in inspect.signature(function).parameters, label
            checked.append(label)
    assert {*ENTRIES, *HELPERS, "CakeMHCNondetPrepareFunction.forward"} <= set(checked)
    assert not hasattr(pkg, "_require_nondeterministic")
    assert "_require_nondeterministic" not in Path(pkg.__file__).read_text()


@pytest.mark.cpu_only
def test_facade_forwards_every_entry_point_to_the_package() -> None:
    """The top-level module ``flashinfer.cake_mhc_nondet_train`` (read as source, not imported) defines its entry
    points and the private ``_pkg`` / ``__getattr__`` / ``__dir__`` only, each entry point with this package's
    parameters (names, kinds and defaults; the ``hidden`` default is the module's ``_HIDDEN``, this package's
    ``cake_jit.HIDDEN``).  Each body forwards to this package's function of the same name, read off ``_pkg()``, with
    every argument unchanged.  The package is imported only inside a function (no module-level import of
    ``flashinfer.experimental``), no import names the deterministic package, and every name the module resolves on
    first access (``_LAZY``) is an exported name of this package."""
    path = REPO / FACADE
    if not path.is_file():
        pytest.skip(f"{FACADE} is not beside this test (installed package)")
    for node, in_function in _imports(path):
        names = [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        assert not [n for n in names if "cake_mhc_det_train" in n], (node.lineno, names)
        # The dotted path of each imported object: ``experimental`` as a path component, not as a substring of a
        # name (``from .api_logging import flashinfer_experimental_api`` imports nothing of the package).
        if isinstance(node, ast.ImportFrom):
            paths = [f"{node.module or ''}.{alias.name}" for alias in node.names]
        else:
            paths = [alias.name for alias in node.names]
        if [p for p in paths if "experimental" in p.split(".")]:
            assert in_function, (node.lineno, paths)
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in ("_HIDDEN", "_LAZY")
    }
    assert constants["_HIDDEN"] == nd.HIDDEN
    assert set(constants["_LAZY"]) <= set(pkg.__all__), constants["_LAZY"]
    functions = _functions(path)
    want = {*FACADE_ENTRIES, "_pkg", "__getattr__", "__dir__"}
    assert set(functions) == want, sorted(set(functions) ^ want)

    def default(node):
        if node is None:
            return inspect.Parameter.empty
        if isinstance(node, ast.Name):
            return constants[node.id]
        return ast.literal_eval(node)

    for name in FACADE_ENTRIES:
        args = functions[name].args
        assert not (args.posonlyargs or args.vararg or args.kwarg), name
        positional_defaults = [None] * (len(args.args) - len(args.defaults))
        parameters = [
            (arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default(node))
            for arg, node in zip(
                args.args, positional_defaults + list(args.defaults), strict=True
            )
        ] + [
            (arg.arg, inspect.Parameter.KEYWORD_ONLY, default(node))
            for arg, node in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        ]
        expected = _shape(inspect.signature(getattr(pkg, name)).parameters.values())
        forwarded = [parameter[0] for parameter in expected]
        body = _body(functions[name])
        assert len(body) == 1, name
        assert parameters == expected, name
        returned = body[-1]
        assert isinstance(returned, ast.Return), name
        call = returned.value
        assert isinstance(call, ast.Call), name
        assert ast.dump(call.func) == _expression(f"_pkg().{name}"), name
        assert all(isinstance(arg, ast.Name) for arg in call.args), name
        for keyword in call.keywords:
            assert isinstance(keyword.value, ast.Name), (name, keyword.arg)
            assert keyword.value.id == keyword.arg, (name, keyword.arg)
        passed = [arg.id for arg in call.args]
        assert passed == forwarded[: len(passed)], name
        passed += [keyword.arg for keyword in call.keywords]
        assert sorted(passed) == sorted(forwarded), name


def _cpu_calls(**keywords) -> dict:
    """One call per entry point on CPU operands of 8 tokens, each with ``keywords`` added."""
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
        "cake_mhc_nondet_train_forward_prepare": lambda: pkg.cake_mhc_nondet_train_forward_prepare(X, fn, scale, base, **keywords),
        "cake_mhc_nondet_train_forward_stats": lambda: pkg.cake_mhc_nondet_train_forward_stats(X, fn, scale, base, **keywords),
        "cake_mhc_nondet_train_forward_finish": lambda: pkg.cake_mhc_nondet_train_forward_finish(X, B, stats, **keywords),
        "cake_mhc_nondet_train_backward_prepare": lambda: pkg.cake_mhc_nondet_train_backward_prepare(X, B, X, stats, **keywords),
        "cake_mhc_nondet_train_backward_finish": lambda: pkg.cake_mhc_nondet_train_backward_finish(
            X, X, dU, B, fn, scale, base, stats, gstats, **keywords
        ),
        "cake_mhc_nondet_train_prepare": lambda: pkg.cake_mhc_nondet_train_prepare(X, fn, scale, base, **keywords),
        "cake_mhc_nondet_train_finish": lambda: pkg.cake_mhc_nondet_train_finish(X, B, object(), **keywords),
        "cake_mhc_nondet_train_forward_workspace_bytes": lambda: pkg.cake_mhc_nondet_train_forward_workspace_bytes(T, cpu, **keywords),
        "cake_mhc_nondet_train_backward_workspace_bytes": lambda: pkg.cake_mhc_nondet_train_backward_workspace_bytes(T, cpu, **keywords),
        "cake_mhc_nondet_train_forward_workspace": lambda: pkg.cake_mhc_nondet_train_forward_workspace(T, cpu, **keywords),
        "cake_mhc_nondet_train_backward_workspace": lambda: pkg.cake_mhc_nondet_train_backward_workspace(T, cpu, **keywords),
    }  # fmt: skip


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", (False, True, None), ids=repr)
def test_entry_points_refuse_a_deterministic_keyword_first(value) -> None:
    """A ``deterministic`` keyword, whatever its value, is the interpreter's ``TypeError`` (an unexpected keyword
    argument) on every entry point and helper, before any operand, device, handle or route check -- also with
    PyTorch's deterministic algorithms on, with no alert (warnings are errors)."""
    calls = _cpu_calls(deterministic=value)
    assert set(calls) == set(ENTRIES)
    calls.update(
        is_cake_mhc_nondet_train_supported=lambda: (
            pkg.is_cake_mhc_nondet_train_supported(
                torch.device("cpu"), deterministic=value
            )
        ),
        cake_mhc_nondet_train_reset_workspace=lambda: (
            pkg.cake_mhc_nondet_train_reset_workspace(None, deterministic=value)
        ),
    )
    for torch_mode in ((False, False), (True, False), (True, True)):
        with (
            _torch_deterministic(torch_mode[0], warn_only=torch_mode[1]),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            for call in calls.values():
                with pytest.raises(TypeError, match=_NO_MODE_KEYWORD):
                    call()


@pytest.mark.cpu_only
def test_every_entry_point_has_a_deterministic_counterpart() -> None:
    """The two modes share their verbs: every entry point of this package (``is_..._supported`` included) has the
    deterministic package's entry point of the same verb (``cake_mhc_det_train_<verb>``) with the same parameters."""
    det_package = cake_package("cake.cake_mhc_det_train")
    for entry in FACADE_ENTRIES:
        det_entry = entry.replace(NONDET_API, DET_API, 1)
        assert det_entry != entry and det_entry in det_package.__all__, entry
        assert _shape(
            inspect.signature(getattr(pkg, entry)).parameters.values()
        ) == _shape(
            inspect.signature(getattr(det_package, det_entry)).parameters.values()
        ), entry


def _called_names(path: Path) -> list[tuple[int, str]]:
    """``(line, name)`` of every call in ``path``: the called function's name, or the attribute of a called method."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                found.append((node.lineno, func.id))
            elif isinstance(func, ast.Attribute):
                found.append((node.lineno, func.attr))
    return found


@pytest.mark.cpu_only
def test_no_call_raises_the_determinism_alert() -> None:
    """Every output of the package repeats bit for bit on one device, so nothing alerts under PyTorch's deterministic
    algorithms: no source file of the package (nor the top-level module, where it is beside this test) calls an
    alert (a function or method whose name holds ``alert``, as PyTorch's ``_alert_not_deterministic`` and a package
    helper around it would) or reads the mode (``are_deterministic_algorithms_enabled``,
    ``is_deterministic_algorithms_warn_only_enabled``)."""
    paths = sorted(PACKAGE_DIR.glob("*.py"))
    assert PACKAGE_DIR / "__init__.py" in paths
    if (REPO / FACADE).is_file():
        paths.append(REPO / FACADE)
    reads = (
        "are_deterministic_algorithms_enabled",
        "is_deterministic_algorithms_warn_only_enabled",
    )
    for path in paths:
        calls = _called_names(path)
        assert not [(line, name) for line, name in calls if "alert" in name.lower()], (
            path
        )
        assert not [(line, name) for line, name in calls if name in reads], path


@pytest.mark.cpu_only
def test_no_entry_alerts_under_deterministic_algorithms(monkeypatch) -> None:
    """Under PyTorch's deterministic algorithms (strict and ``warn_only=True``, warnings are errors) every entry
    point, ``backward_finish`` included, refuses the CPU operands here with its operand error, no alert, and
    launches nothing (the GPU suite checks that every entry point runs and gives the bits it gives with the mode off);
    ``cake_mhc_nondet_train_finish`` refuses the handle its prepare did not make with a ``TypeError`` that names the
    deterministic finish."""
    launched = []
    monkeypatch.setattr(pkg, "_launch_route", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(pkg, "_run_route", lambda *a, **k: launched.append(a))
    calls = _cpu_calls()
    assert "cake_mhc_nondet_train_backward_finish" in calls
    for warn_only in (False, True):
        with _torch_deterministic(True, warn_only=warn_only), warnings.catch_warnings():
            warnings.simplefilter("error")
            for name, call in calls.items():
                with pytest.raises((ValueError, TypeError)) as info:
                    call()
                assert _ALERT not in str(info.value), (name, warn_only)
                if name == "cake_mhc_nondet_train_finish":
                    assert info.type is TypeError, warn_only
                    assert str(info.value) == _HANDLE_REFUSAL, warn_only
    assert launched == []


# ----------------------------------------------------------------------------------------------------------------
# Wiring and the build manifest


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    """The test-helper key, ``flashinfer.experimental.cake`` (a docstring only), the package data, the ignored
    ``aot/`` directory and the pre-commit exclude of the generated sources; no key, package data, exclude or module
    is left at the former paths (``flashinfer.experimental.cake.cake_mhc_train_backend`` and its
    ``cake_mhc_train`` module, the 0.7.3 ``flashinfer.cake_mhc_train_backend``)."""
    assert PACKAGES[KEY] == PACKAGE_NAME == pkg.__name__
    for former in FORMER_PACKAGES:
        assert former not in PACKAGES.values(), former
    assert not {"cake_mhc_train", "cake.cake_mhc_train"} & set(PACKAGES)
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
    for former in FORMER_PACKAGES:
        assert former not in package_data, former
    assert "aot/" in (PACKAGE_DIR / ".gitignore").read_text().split()
    pre_commit = (REPO / ".pre-commit-config.yaml").read_text()
    # pre-commit skips every generated source of the package (a per-package entry or the generic #8928 rule), still
    # checks its Python modules; the former tree holds no sources (the shared exclude may still list its path)
    exclude = re.compile(
        next(ln for ln in pre_commit.splitlines() if ln.startswith("exclude:"))
        .split(":", 1)[1]
        .strip()
    )
    root, package_dir = REPO.resolve(), PACKAGE_DIR.resolve()
    sources = [
        p.relative_to(root).as_posix()
        for p in sorted((package_dir / "csrc").rglob("*"))
        if p.is_file()
    ]
    assert sources and all(exclude.search(p) for p in sources), [
        p for p in sources if not exclude.search(p)
    ]
    assert not exclude.search(
        (package_dir / "cake_jit.py").relative_to(root).as_posix()
    )
    assert not list(
        (REPO / "flashinfer/experimental/cake/cake_mhc_train_backend/csrc").glob("*")
    )
    for former in FORMER_PACKAGES:
        directory = REPO.joinpath(*former.split("."))
        assert not list(directory.glob("*.py")), former
    assert not (REPO / "flashinfer/experimental/cake/cake_mhc_train.py").exists()


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
    """The build-manifest checker lists this package once (generic registry walker over ``cake_jit.py``) and no former
    path (``FORMER_PACKAGES``); its vendored gate finds the package's suite key; the in-tree checks report nothing."""
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
    assert not [p for p in checker.PACKAGES if p.name in FORMER_PACKAGES]
    assert checker.suite_key(REPO, entry) == KEY
    assert checker.run_in_tree(REPO, [entry]) == []
