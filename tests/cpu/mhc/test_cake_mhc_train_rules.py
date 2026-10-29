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

Host-rule tests of the Cake four-stream mHC training package (no GPU needed).

The registry ``cake_jit.py`` carries the integer host rules of the kernels
(``k1_units`` with its per-architecture P_K band, ``k7_chunks``, ``k6_grid``,
``k2_grid``, the one-trip row grids ``k3_grid`` / ``k4_grid``, the backward prepare's
``k5_program`` / ``k5_grid`` with its per-architecture CTA-form bands ``K5_FORM_BANDS``
and persistent bands ``K5_GRID_BANDS``,
the design's persistent ``row_grid``, the token padding and the workspace regions)
and the program dispatch of record (``fwd_map_program``, ``fwd_coef_program``
and ``bwd_token_program``: which mapping-pass, coefficient-pass and token-owner
form a call launches, by token count, SM count and architecture;
``bwd_reduce_program`` / ``bwd_reduce_grid``: the reduction form and its fixed
grid per token count and architecture; ``fn_split_program`` the fn split's form per
architecture and route (``want_t`` 0 = the forward operand layout, 1 = the
transposed layout of the backward) with its launch grid ``fn_split_grid``, and ``bwd_slab_program`` the
column-slab form per token count and architecture -- its token-count bands
``BWD_SLAB_BANDS`` ahead of the per-architecture ``BWD_SLAB_PROGRAM``) as plain Python.  The same rules exist in
the kernels' own test harness; :data:`RULES_ROWS` is that harness's table of
record -- one row per (token count, board) over the 28 token counts the
contract names (the 16 correctness values and the 13 performance rows) and the
four (architecture, SM count) boards of the target products -- and
:data:`DISPATCH_EXPECTED` its dispatch at every band boundary of
:data:`DISPATCH_T` on those boards; the registry must reproduce every row
bitwise.  The literal tables of this module (the libraries per kernel, the
architectures that route each library, the libraries launched under several
records, the record scalars, the master's per-architecture tables) are
generated together with the registry from the kernel-tests reference master.
The module also checks the registry's literal tables (the exporter reads them
with ``ast.literal_eval``), that every forward call selects exactly one
fn-split, one mapping-pass and one coefficient-pass record and every backward
call exactly one fn-split, one token-owner, one column-slab and one reduction
record, that the selection follows the rules of record, the launch grids of
every grid-rule kind on those boards, the workspace layout (256-byte regions,
the monotone size envelope the public API returns), the launch facts of record
(block, dynamic shared memory), that programmatic dependent launch is
enabled exactly on the modules whose kernels wait for the preceding grid --
per call, inside the measured token-count bands ``PDL_OFF_BANDS``, on the
dependents of an early-releasing kernel (``PDL_EDGES``) -- and that the
backward prepare's CTA form of record (``k5_program``) selects the record of
its library.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import cake_package

loader = cake_package("cake_mhc_train", "cake_jit")

SM_COUNTS = (148, 152, 208, 216)
BOARDS = (("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 208), ("sm_107a", 216))
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# Program key -> library per kernel, the libraries the registry ships (generated from the records).
SPLIT_LIBRARY = {
    "b16": "cake_mhc_train_fn_split",
    "tv128": "cake_mhc_train_fn_split__tv128",
    "tv128_t128": "cake_mhc_train_fn_split__tv128_t128",
}
MAP_LIBRARY = {
    "ss_st4": "cake_mhc_train_fwd_map",
    "nsx": "cake_mhc_train_fwd_map__nsx",
    "nsx_st5_mb1": "cake_mhc_train_fwd_map__nsx_st5_mb1",
    "nsx_g2_ol2l": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
    "ns96_g2_ol2l": "cake_mhc_train_fwd_map__ns96_g2_ol2l",
}
COEF_LIBRARY = {
    "qe_mb10": "cake_mhc_train_fwd_coef__qe_mb10",
    "qe_mb10_pu4": "cake_mhc_train_fwd_coef__qe_mb10_pu4",
    "st8_lp2": "cake_mhc_train_fwd_coef__st8_lp2",
    "lp2_w8_adj": "cake_mhc_train_fwd_coef__lp2_w8_adj",
    "st16_w4_tree": "cake_mhc_train_fwd_coef__st16_w4_tree",
    "qe_mb10_tree": "cake_mhc_train_fwd_coef__qe_mb10_tree",
    "st16_lp2_as_el_qd": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
    "st8_lp2_as_el_qd": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd",
    "pfl2_qmt": "cake_mhc_train_fwd_coef__pfl2_qmt",
    "g2_el": "cake_mhc_train_fwd_coef__g2_el",
    "st8_lp2_tree_asel_qd": "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd",
    "lp2_w8_adj_qe": "cake_mhc_train_fwd_coef__lp2_w8_adj_qe",
}
TOKEN_LIBRARY = {
    "dot_ao8": "cake_mhc_train_bwd_token__dot_ao8",
    "ao22_rae7_pd1_ntr": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
    "ao15_rae8_tma_ntr": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
    "ao22_rae7_rf3_pd1_l2l_ntr": "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr",
    "ao22_rae7_pd1_tma_ntr": "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr",
    "dot_ao8_l2l": "cake_mhc_train_bwd_token__dot_ao8_l2l",
}
# The run-time-count records (select rule ``run_time_iters``): record -> the kernel it stands in for at every
# Sinkhorn count but the compiled one (the count is their ``sinkhorn_iters`` argument, at most SINKHORN_ITERS_MAX).
RUN_TIME_ITERS = {
    "cake_mhc_train_fwd_coef__rt64": "cake_mhc_train_fwd_coef",
    "cake_mhc_train_bwd_token__rt64": "cake_mhc_train_bwd_token",
}
SLAB_LIBRARY = {
    "xrsg": "cake_mhc_train_bwd_slab__xrsg",
    "dynr": "cake_mhc_train_bwd_slab__dynr",
}
REDUCE_LIBRARY = {
    "pt64": "cake_mhc_train_bwd_reduce__pt64",
    "pt32": "cake_mhc_train_bwd_reduce__pt32",
    "wpu4": "cake_mhc_train_bwd_reduce__wpu4",
    "s27_wcvt_spred": "cake_mhc_train_bwd_reduce__s27_wcvt_spred",
    "s27_wcvt_fadd_fshfl": "cake_mhc_train_bwd_reduce__s27_wcvt_fadd_fshfl",
    "wpu4_fadd": "cake_mhc_train_bwd_reduce__wpu4_fadd",
    "wpu4_spred": "cake_mhc_train_bwd_reduce__wpu4_spred",
}
COEF_TOKENS_PER_CTA = {
    "qe_mb10": 8,
    "qe_mb10_pu4": 8,
    "st8_lp2": 8,
    "lp2_w8_adj": 8,
    "st16_w4_tree": 4,
    "qe_mb10_tree": 8,
    "st16_lp2_as_el_qd": 8,
    "st8_lp2_as_el_qd": 8,
    "pfl2_qmt": 8,
    "g2_el": 16,
    "st8_lp2_tree_asel_qd": 8,
    "lp2_w8_adj_qe": 8,
}
REDUCE_GRID = {
    "pt64": 1563,
    "pt32": 3099,
    "wpu4": 411,
    "s27_wcvt_spred": 411,
    "s27_wcvt_fadd_fshfl": 411,
    "wpu4_fadd": 411,
    "wpu4_spred": 411,
}
FINISH_LIBRARY = {
    "production": "cake_mhc_train_fwd_finish",
    "r2": "cake_mhc_train_fwd_finish__r2",
}
# Library -> the architectures whose routes name a record of it (generated from the records).
LIBRARY_ARCHES = {
    "cake_mhc_train_bwd_prepare": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_prepare__t128": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_reduce__pt32": ("sm_103a",),
    "cake_mhc_train_bwd_reduce__pt64": ("sm_103a",),
    "cake_mhc_train_bwd_reduce__s27_wcvt_fadd_fshfl": ("sm_100a",),
    "cake_mhc_train_bwd_reduce__s27_wcvt_spred": ("sm_107a",),
    "cake_mhc_train_bwd_reduce__wpu4": ("sm_107a",),
    "cake_mhc_train_bwd_reduce__wpu4_fadd": ("sm_100a", "sm_107a"),
    "cake_mhc_train_bwd_reduce__wpu4_spred": ("sm_100a",),
    "cake_mhc_train_bwd_slab__dynr": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab__xrsg": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr": ("sm_100a", "sm_107a"),
    "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr": ("sm_107a",),
    "cake_mhc_train_bwd_token__dot_ao8": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_token__dot_ao8_l2l": ("sm_107a",),
    "cake_mhc_train_bwd_token__rt64": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fn_split": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fn_split__tv128": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fn_split__tv128_t128": ("sm_107a",),
    "cake_mhc_train_fwd_coef__g2_el": ("sm_107a",),
    "cake_mhc_train_fwd_coef__lp2_w8_adj": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_coef__lp2_w8_adj_qe": ("sm_107a",),
    "cake_mhc_train_fwd_coef__pfl2_qmt": ("sm_107a",),
    "cake_mhc_train_fwd_coef__qe_mb10": ("sm_103a",),
    "cake_mhc_train_fwd_coef__qe_mb10_pu4": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_coef__qe_mb10_tree": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__rt64": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__st16_w4_tree": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__st8_lp2": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd": ("sm_100a",),
    "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd": ("sm_107a",),
    "cake_mhc_train_fwd_collapse": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_finish": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_finish__r2": ("sm_107a",),
    "cake_mhc_train_fwd_map": ("sm_107a",),
    "cake_mhc_train_fwd_map__ns96_g2_ol2l": ("sm_100a",),
    "cake_mhc_train_fwd_map__nsx": ("sm_103a",),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_map__nsx_st5_mb1": ("sm_100a", "sm_103a", "sm_107a"),
}
# Libraries launched under several records (one record per selection rule, in registry order).
MULTI_RECORDS = {
    "cake_mhc_train_fn_split__tv128": ["cake_mhc_train_fn_split__tv128__t"],
    "cake_mhc_train_fn_split__tv128_t128": ["cake_mhc_train_fn_split__tv128_t128__t"],
    "cake_mhc_train_fwd_map__nsx_st5_mb1": [
        "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "cake_mhc_train_fwd_map__nsx_st5_mb1__a3",
        "cake_mhc_train_fwd_map__nsx_st5_mb1__a7",
    ],
    "cake_mhc_train_fwd_map__nsx_g2_ol2l": [
        "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "cake_mhc_train_fwd_map__nsx_g2_ol2l__a3",
        "cake_mhc_train_fwd_map__nsx_g2_ol2l__a7",
    ],
    "cake_mhc_train_fwd_map": ["cake_mhc_train_fwd_map__hi"],
    "cake_mhc_train_bwd_prepare": [
        "cake_mhc_train_bwd_prepare__g2sm_le1024",
        "cake_mhc_train_bwd_prepare__gt8192",
    ],
    "cake_mhc_train_bwd_prepare__t128": [
        "cake_mhc_train_bwd_prepare__t128",
        "cake_mhc_train_bwd_prepare__t128__a3",
    ],
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr": [
        "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a3",
        "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a7",
    ],
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr": [
        "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a3",
        "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a7",
    ],
    "cake_mhc_train_bwd_token__dot_ao8": [
        "cake_mhc_train_bwd_token__dot_ao8",
        "cake_mhc_train_bwd_token__dot_ao8__a3",
    ],
    "cake_mhc_train_bwd_slab__xrsg": [
        "cake_mhc_train_bwd_slab__xrsg",
        "cake_mhc_train_bwd_slab__xrsg__a7",
    ],
}
N_RECORDS, N_LIBRARIES = 54, 42
RECORD_SCALARS = {
    "cake_mhc_train_fn_split": {"want_t": 0},
    "cake_mhc_train_fn_split__tv128__t": {"want_t": 1},
    "cake_mhc_train_fn_split__tv128_t128__t": {"want_t": 1},
}
# The reference master's per-architecture tables of record (generated from the master text).
TABLES = {
    "PK_BANDS": {"sm_100a": (), "sm_103a": ((8192, 16384, 16),), "sm_107a": ()},
    "FN_SPLIT_PROGRAM": {
        "sm_100a": ("b16", "tv128"),
        "sm_103a": ("b16", "tv128"),
        "sm_107a": ("b16", "tv128_t128"),
    },
    "FN_SPLIT_GRID": {"b16": 256, "tv128": 256, "tv128_t128": 512},
    "FWD_MAP_PRODUCTION": "ss_st4",
    "FWD_MAP_BANDS": {
        "sm_100a": ((0, 8192, "ns96_g2_ol2l"), (8192, None, "nsx_g2_ol2l")),
        "sm_103a": ((0, 8191, "nsx"), (8191, None, "nsx_g2_ol2l")),
        "sm_107a": ((0, 32768, "nsx_g2_ol2l"),),
    },
    "FWD_MAP_FIT": {
        "sm_100a": ("ns96_g2_ol2l", "nsx_st5_mb1"),
        "sm_103a": ("nsx", "nsx_st5_mb1"),
        "sm_107a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
    },
    "FWD_COEF_REGIMES": {"sm_100a": "round7", "sm_103a": "round5", "sm_107a": "round9"},
    "BWD_TOKEN_PROGRAM": {
        "sm_100a": "dot_ao8",
        "sm_103a": "dot_ao8",
        "sm_107a": "dot_ao8_l2l",
    },
    "BWD_TOKEN_REGIMES": {
        "sm_100a": (("ao22_rae7_pd1_ntr", 0, 32), ("ao15_rae8_tma_ntr", 49, 56)),
        "sm_103a": (("ao22_rae7_pd1_ntr", 0, 32), ("ao15_rae8_tma_ntr", 49, 56)),
        "sm_107a": (("ao22_rae7_pd1_ntr", 0, 32), ("ao15_rae8_tma_ntr", 49, 56)),
    },
    "BWD_SLAB_PROGRAM": {"sm_100a": "xrsg", "sm_103a": "xrsg", "sm_107a": "xrsg"},
    "BWD_SLAB_BANDS": {
        "sm_100a": ((4096, 8192, "dynr"),),
        "sm_103a": ((0, 4096, "dynr"),),
        "sm_107a": (),
    },
    "BWD_REDUCE_PROGRAM": {
        "sm_100a": "s27_wcvt",
        "sm_103a": "pt64",
        "sm_107a": "s27_wcvt",
    },
    "BWD_REDUCE_REGIMES": {
        "sm_100a": (),
        "sm_103a": (("pt32", 4095, None),),
        "sm_107a": (("s27_wcvt_spred", 0, 1024), ("wpu4", 1024, None)),
    },
    "BWD_REDUCE_GRID": {
        "f64": 385,
        "s27": 411,
        "s27_wcvt": 411,
        "pt64": 1563,
        "pt32": 3099,
        "wpu4": 411,
        "wpu4_pfn": 411,
        "s27_wcvt_spred": 411,
        "s27_wcvt_fadd_fshfl": 411,
        "wpu4_fadd": 411,
        "wpu4_spred": 411,
    },
    "BWD_REDUCE_PARTS": {
        "sm_100a": {
            "s27_wcvt": (
                ("s27_wcvt_fadd_fshfl", 0, 5),
                ("wpu4_spred", 6, 12),
                ("wpu4_fadd", 13, 19),
                ("wpu4_spred", 20, None),
            )
        },
        "sm_103a": {},
        "sm_107a": {"wpu4": (("wpu4_fadd", 15, None),)},
    },
    "BWD_TOKEN_BANDS": {
        "sm_100a": ((1024, 8192, "ao22_rae7_pd1_tma_ntr"),),
        "sm_103a": (),
        "sm_107a": (
            (0, 1024, "ao22_rae7_rf3_pd1_l2l_ntr"),
            (1024, 4096, "ao22_rae7_pd1_tma_ntr"),
        ),
    },
    "K5_GRID_BANDS": {
        "sm_100a": ((0, 1024, 2),),
        "sm_103a": ((0, 4096, 2),),
        "sm_107a": (),
    },
    "FWD_FINISH_PROGRAM": {
        "sm_100a": "production",
        "sm_103a": "production",
        "sm_107a": "r2",
    },
    "FWD_FINISH_ROWS": {"production": 4, "r2": 2},
    "K5_FORM_BANDS": {
        "sm_100a": ((1024, 8192, "t128"),),
        "sm_103a": ((0, 8192, "t128"),),
        "sm_107a": ((0, 8192, "t128"),),
    },
    "K5_FORM_ROWS": {"production": 2, "t128": 1},
}
# The token counts that exercise every band boundary of the dispatch at the SM counts of record (the kernel-tests
# rules test's DISPATCH_T).
DISPATCH_T = (
    1,
    2,
    3,
    7,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    1000,
    1024,
    1152,
    1153,
    1176,
    1177,
    1208,
    1209,
    1656,
    1657,
    1664,
    1665,
    1720,
    1721,
    1792,
    1793,
    2049,
    2304,
    2305,
    2432,
    2433,
    2560,
    2561,
    3328,
    3329,
    3456,
    3457,
    4095,
    4096,
    5328,
    5329,
    5472,
    5473,
    7488,
    7489,
    7776,
    7777,
    8191,
    8192,
    8193,
    9472,
    9473,
    9728,
    9729,
    11840,
    11841,
    12160,
    12161,
    13312,
    13313,
    13824,
    13825,
    14208,
    14209,
    14592,
    14593,
    14928,
    15288,
    15392,
    15393,
    15566,
    15590,
    15808,
    15809,
    15984,
    15985,
    16114,
    16132,
    16164,
    16263,
    16384,
    16385,
    16416,
    16417,
    16576,
    16577,
    16640,
    16641,
    17024,
    17025,
    17280,
    17281,
    19968,
    19969,
    20001,
    20736,
    20737,
    21632,
    21633,
    22464,
    22465,
    23296,
    23297,
    23328,
    23329,
    24192,
    24193,
    32767,
    32768,
    32769,
    65536,
)
# (T, arch, SM count, P_K, K1 units, K2 program, K1 program, R6, K6 program, K7 program, K8 program, K8 grid) at every boundary T of DISPATCH_T on the boards of record -- the kernel-tests reference master's dispatch
DISPATCH_EXPECTED = (
    (1, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (1, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (2, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (2, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (2, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (3, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (3, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (3, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (3, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (7, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (7, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (7, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (7, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (63, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (63, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (63, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (63, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (64, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (64, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (64, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (64, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (65, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (65, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (65, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (65, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (127, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (127, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (127, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (127, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (128, 'sm_100a', 148, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (128, 'sm_103a', 152, 32, 32, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (128, 'sm_107a', 208, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (128, 'sm_107a', 216, 32, 32, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (129, 'sm_100a', 148, 32, 64, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (129, 'sm_103a', 152, 32, 64, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (129, 'sm_107a', 208, 32, 64, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (129, 'sm_107a', 216, 32, 64, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (255, 'sm_100a', 148, 32, 64, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (255, 'sm_103a', 152, 32, 64, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (255, 'sm_107a', 208, 32, 64, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (255, 'sm_107a', 216, 32, 64, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (1000, 'sm_100a', 148, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1000, 'sm_103a', 152, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1000, 'sm_107a', 208, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (1000, 'sm_107a', 216, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (1024, 'sm_100a', 148, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1024, 'sm_103a', 152, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1024, 'sm_107a', 208, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (1024, 'sm_107a', 216, 16, 128, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411),
    (1152, 'sm_100a', 148, 16, 144, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1152, 'sm_103a', 152, 16, 144, 'nsx_st5_mb1', 'st16_w4_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1152, 'sm_107a', 208, 16, 144, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1152, 'sm_107a', 216, 16, 144, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1153, 'sm_100a', 148, 32, 320, 'ns96_g2_ol2l', 'st16_w4_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1153, 'sm_103a', 152, 32, 320, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1153, 'sm_107a', 208, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1153, 'sm_107a', 216, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1176, 'sm_100a', 148, 32, 320, 'ns96_g2_ol2l', 'st16_w4_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1176, 'sm_103a', 152, 32, 320, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1176, 'sm_107a', 208, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1176, 'sm_107a', 216, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1177, 'sm_100a', 148, 32, 320, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1177, 'sm_103a', 152, 32, 320, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1177, 'sm_107a', 208, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1177, 'sm_107a', 216, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1208, 'sm_100a', 148, 32, 320, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1208, 'sm_103a', 152, 32, 320, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1208, 'sm_107a', 208, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1208, 'sm_107a', 216, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1209, 'sm_100a', 148, 32, 320, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1209, 'sm_103a', 152, 32, 320, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1209, 'sm_107a', 208, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1209, 'sm_107a', 216, 16, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1656, 'sm_100a', 148, 32, 416, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1656, 'sm_103a', 152, 32, 416, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1656, 'sm_107a', 208, 16, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1656, 'sm_107a', 216, 16, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1657, 'sm_100a', 148, 32, 416, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1657, 'sm_103a', 152, 32, 416, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1657, 'sm_107a', 208, 16, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1657, 'sm_107a', 216, 16, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1664, 'sm_100a', 148, 32, 416, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1664, 'sm_103a', 152, 32, 416, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1664, 'sm_107a', 208, 16, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1664, 'sm_107a', 216, 16, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1665, 'sm_100a', 148, 8, 112, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (1665, 'sm_103a', 152, 32, 448, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1665, 'sm_107a', 208, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1665, 'sm_107a', 216, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1720, 'sm_100a', 148, 8, 112, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (1720, 'sm_103a', 152, 32, 448, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1720, 'sm_107a', 208, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1720, 'sm_107a', 216, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1721, 'sm_100a', 148, 8, 112, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (1721, 'sm_103a', 152, 32, 448, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1721, 'sm_107a', 208, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1721, 'sm_107a', 216, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1792, 'sm_100a', 148, 8, 112, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1792, 'sm_103a', 152, 32, 448, 'nsx', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1792, 'sm_107a', 208, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1792, 'sm_107a', 216, 32, 448, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1793, 'sm_100a', 148, 8, 120, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (1793, 'sm_103a', 152, 8, 120, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (1793, 'sm_107a', 208, 32, 480, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (1793, 'sm_107a', 216, 32, 480, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2049, 'sm_100a', 148, 8, 136, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (2049, 'sm_103a', 152, 8, 136, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2049, 'sm_107a', 208, 32, 544, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2049, 'sm_107a', 216, 32, 544, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2304, 'sm_100a', 148, 8, 144, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (2304, 'sm_103a', 152, 8, 144, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2304, 'sm_107a', 208, 32, 576, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2304, 'sm_107a', 216, 32, 576, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2305, 'sm_100a', 148, 32, 608, 'ns96_g2_ol2l', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (2305, 'sm_103a', 152, 8, 152, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2305, 'sm_107a', 208, 32, 608, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2305, 'sm_107a', 216, 32, 608, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2432, 'sm_100a', 148, 32, 608, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (2432, 'sm_103a', 152, 8, 152, 'nsx_st5_mb1', 'qe_mb10_tree', 8, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2432, 'sm_107a', 208, 32, 608, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2432, 'sm_107a', 216, 32, 608, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2433, 'sm_100a', 148, 32, 640, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (2433, 'sm_103a', 152, 32, 640, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2433, 'sm_107a', 208, 8, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2433, 'sm_107a', 216, 32, 640, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2560, 'sm_100a', 148, 32, 640, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (2560, 'sm_103a', 152, 32, 640, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2560, 'sm_107a', 208, 8, 160, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2560, 'sm_107a', 216, 32, 640, 'nsx_g2_ol2l', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2561, 'sm_100a', 148, 32, 672, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411),
    (2561, 'sm_103a', 152, 32, 672, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (2561, 'sm_107a', 208, 8, 168, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (2561, 'sm_107a', 216, 8, 168, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3328, 'sm_100a', 148, 16, 416, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (3328, 'sm_103a', 152, 16, 416, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (3328, 'sm_107a', 208, 8, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3328, 'sm_107a', 216, 8, 208, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3329, 'sm_100a', 148, 16, 432, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (3329, 'sm_103a', 152, 16, 432, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (3329, 'sm_107a', 208, 32, 864, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3329, 'sm_107a', 216, 8, 216, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3456, 'sm_100a', 148, 16, 432, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (3456, 'sm_103a', 152, 16, 432, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (3456, 'sm_107a', 208, 32, 864, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3456, 'sm_107a', 216, 8, 216, 'nsx_st5_mb1', 'pfl2_qmt', 8, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3457, 'sm_100a', 148, 8, 224, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (3457, 'sm_103a', 152, 16, 448, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (3457, 'sm_107a', 208, 32, 896, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (3457, 'sm_107a', 216, 32, 896, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (4095, 'sm_100a', 148, 8, 256, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (4095, 'sm_103a', 152, 8, 256, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563),
    (4095, 'sm_107a', 208, 32, 1024, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (4095, 'sm_107a', 216, 32, 1024, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (4096, 'sm_100a', 148, 8, 256, 'ns96_g2_ol2l', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (4096, 'sm_103a', 152, 8, 256, 'nsx', 'qe_mb10_tree', 16, 'ao22_rae7_pd1_ntr', 'dynr', 'pt32', 3099),
    (4096, 'sm_107a', 208, 32, 1024, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (4096, 'sm_107a', 216, 32, 1024, 'nsx_g2_ol2l', 'pfl2_qmt', 16, 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411),
    (5328, 'sm_100a', 148, 16, 672, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (5328, 'sm_103a', 152, 16, 672, 'nsx', 'qe_mb10_tree', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (5328, 'sm_107a', 208, 8, 336, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5328, 'sm_107a', 216, 8, 336, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5329, 'sm_100a', 148, 16, 672, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (5329, 'sm_103a', 152, 16, 672, 'nsx', 'qe_mb10_tree', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (5329, 'sm_107a', 208, 8, 336, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5329, 'sm_107a', 216, 8, 336, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5472, 'sm_100a', 148, 16, 688, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (5472, 'sm_103a', 152, 16, 688, 'nsx', 'qe_mb10_tree', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (5472, 'sm_107a', 208, 8, 344, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5472, 'sm_107a', 216, 8, 344, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5473, 'sm_100a', 148, 16, 688, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (5473, 'sm_103a', 152, 16, 688, 'nsx', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (5473, 'sm_107a', 208, 8, 344, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (5473, 'sm_107a', 216, 8, 344, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 16, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7488, 'sm_100a', 148, 16, 944, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (7488, 'sm_103a', 152, 16, 944, 'nsx', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (7488, 'sm_107a', 208, 16, 944, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7488, 'sm_107a', 216, 16, 944, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7489, 'sm_100a', 148, 16, 944, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (7489, 'sm_103a', 152, 16, 944, 'nsx', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (7489, 'sm_107a', 208, 16, 944, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7489, 'sm_107a', 216, 16, 944, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7776, 'sm_100a', 148, 16, 976, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (7776, 'sm_103a', 152, 16, 976, 'nsx', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (7776, 'sm_107a', 208, 16, 976, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7776, 'sm_107a', 216, 16, 976, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7777, 'sm_100a', 148, 16, 976, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_spred', 411),
    (7777, 'sm_103a', 152, 16, 976, 'nsx', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (7777, 'sm_107a', 208, 16, 976, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (7777, 'sm_107a', 216, 16, 976, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (8191, 'sm_100a', 148, 16, 1024, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_fadd', 411),
    (8191, 'sm_103a', 152, 16, 1024, 'nsx', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (8191, 'sm_107a', 208, 16, 1024, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (8191, 'sm_107a', 216, 16, 1024, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (8192, 'sm_100a', 148, 16, 1024, 'ns96_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_fadd', 411),
    (8192, 'sm_103a', 152, 16, 1024, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (8192, 'sm_107a', 208, 16, 1024, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (8192, 'sm_107a', 216, 16, 1024, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (8193, 'sm_100a', 148, 8, 520, 'nsx_g2_ol2l', 'qe_mb10_tree', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4_spred', 411),
    (8193, 'sm_103a', 152, 16, 1040, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (8193, 'sm_107a', 208, 16, 1040, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (8193, 'sm_107a', 216, 16, 1040, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9472, 'sm_100a', 148, 8, 592, 'nsx_g2_ol2l', 'qe_mb10_tree', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4_spred', 411),
    (9472, 'sm_103a', 152, 16, 1184, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (9472, 'sm_107a', 208, 8, 592, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9472, 'sm_107a', 216, 8, 592, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9473, 'sm_100a', 148, 16, 1200, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (9473, 'sm_103a', 152, 16, 1200, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (9473, 'sm_107a', 208, 8, 600, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9473, 'sm_107a', 216, 8, 600, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9728, 'sm_100a', 148, 16, 1216, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (9728, 'sm_103a', 152, 16, 1216, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099),
    (9728, 'sm_107a', 208, 8, 608, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9728, 'sm_107a', 216, 8, 608, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9729, 'sm_100a', 148, 16, 1232, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (9729, 'sm_103a', 152, 16, 1232, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (9729, 'sm_107a', 208, 8, 616, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (9729, 'sm_107a', 216, 8, 616, 'nsx_g2_ol2l', 'g2_el', 24, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (11840, 'sm_100a', 148, 16, 1488, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (11840, 'sm_103a', 152, 16, 1488, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (11840, 'sm_107a', 208, 8, 744, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (11840, 'sm_107a', 216, 16, 1488, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (11841, 'sm_100a', 148, 16, 1488, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (11841, 'sm_103a', 152, 16, 1488, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (11841, 'sm_107a', 208, 8, 744, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (11841, 'sm_107a', 216, 16, 1488, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (12160, 'sm_100a', 148, 16, 1520, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (12160, 'sm_103a', 152, 16, 1520, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 40, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (12160, 'sm_107a', 208, 8, 760, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (12160, 'sm_107a', 216, 8, 760, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (12161, 'sm_100a', 148, 16, 1536, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (12161, 'sm_103a', 152, 16, 1536, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (12161, 'sm_107a', 208, 8, 768, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (12161, 'sm_107a', 216, 8, 768, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (13312, 'sm_100a', 148, 8, 832, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (13312, 'sm_103a', 152, 16, 1664, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (13312, 'sm_107a', 208, 8, 832, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (13312, 'sm_107a', 216, 8, 832, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (13313, 'sm_100a', 148, 8, 840, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (13313, 'sm_103a', 152, 16, 1680, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (13313, 'sm_107a', 208, 16, 1680, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (13313, 'sm_107a', 216, 8, 840, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411),
    (13824, 'sm_100a', 148, 8, 864, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (13824, 'sm_103a', 152, 16, 1728, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (13824, 'sm_107a', 208, 16, 1728, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (13824, 'sm_107a', 216, 8, 864, 'nsx_g2_ol2l', 'g2_el', 32, 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4_fadd', 411),
    (13825, 'sm_100a', 148, 8, 872, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (13825, 'sm_103a', 152, 16, 1744, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (13825, 'sm_107a', 208, 16, 1744, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (13825, 'sm_107a', 216, 16, 1744, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14208, 'sm_100a', 148, 8, 888, 'nsx_g2_ol2l', 'qe_mb10_pu4', 48, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (14208, 'sm_103a', 152, 16, 1776, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (14208, 'sm_107a', 208, 16, 1776, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (14208, 'sm_107a', 216, 16, 1776, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14209, 'sm_100a', 148, 8, 896, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (14209, 'sm_103a', 152, 16, 1792, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (14209, 'sm_107a', 208, 16, 1792, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (14209, 'sm_107a', 216, 16, 1792, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14592, 'sm_100a', 148, 8, 912, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (14592, 'sm_103a', 152, 16, 1824, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 48, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (14592, 'sm_107a', 208, 16, 1824, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14592, 'sm_107a', 216, 16, 1824, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14593, 'sm_100a', 148, 8, 920, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (14593, 'sm_103a', 152, 16, 1840, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (14593, 'sm_107a', 208, 16, 1840, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14593, 'sm_107a', 216, 16, 1840, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14928, 'sm_100a', 148, 8, 936, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (14928, 'sm_103a', 152, 16, 1872, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (14928, 'sm_107a', 208, 16, 1872, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (14928, 'sm_107a', 216, 16, 1872, 'nsx_g2_ol2l', 'g2_el', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15288, 'sm_100a', 148, 8, 960, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (15288, 'sm_103a', 152, 16, 1920, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15288, 'sm_107a', 208, 8, 960, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (15288, 'sm_107a', 216, 16, 1920, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15392, 'sm_100a', 148, 8, 968, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15392, 'sm_103a', 152, 16, 1936, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15392, 'sm_107a', 208, 8, 968, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15392, 'sm_107a', 216, 16, 1936, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (15393, 'sm_100a', 148, 8, 968, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15393, 'sm_103a', 152, 16, 1936, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15393, 'sm_107a', 208, 8, 968, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15393, 'sm_107a', 216, 16, 1936, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (15566, 'sm_100a', 148, 8, 976, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15566, 'sm_103a', 152, 16, 1952, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15566, 'sm_107a', 208, 8, 976, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15566, 'sm_107a', 216, 8, 976, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15590, 'sm_100a', 148, 8, 976, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15590, 'sm_103a', 152, 16, 1952, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15590, 'sm_107a', 208, 8, 976, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15590, 'sm_107a', 216, 8, 976, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15808, 'sm_100a', 148, 8, 992, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15808, 'sm_103a', 152, 16, 1984, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15808, 'sm_107a', 208, 8, 992, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15808, 'sm_107a', 216, 8, 992, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15809, 'sm_100a', 148, 8, 992, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15809, 'sm_103a', 152, 16, 1984, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15809, 'sm_107a', 208, 8, 992, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15809, 'sm_107a', 216, 8, 992, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15984, 'sm_100a', 148, 8, 1000, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15984, 'sm_103a', 152, 16, 2000, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15984, 'sm_107a', 208, 8, 1000, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15984, 'sm_107a', 216, 8, 1000, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15985, 'sm_100a', 148, 8, 1000, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (15985, 'sm_103a', 152, 16, 2000, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (15985, 'sm_107a', 208, 8, 1000, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (15985, 'sm_107a', 216, 8, 1000, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16114, 'sm_100a', 148, 8, 1008, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16114, 'sm_103a', 152, 16, 2016, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16114, 'sm_107a', 208, 8, 1008, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16114, 'sm_107a', 216, 8, 1008, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16132, 'sm_100a', 148, 8, 1016, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16132, 'sm_103a', 152, 16, 2032, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16132, 'sm_107a', 208, 8, 1016, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16132, 'sm_107a', 216, 8, 1016, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16164, 'sm_100a', 148, 8, 1016, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16164, 'sm_103a', 152, 16, 2032, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16164, 'sm_107a', 208, 8, 1016, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16164, 'sm_107a', 216, 8, 1016, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16263, 'sm_100a', 148, 8, 1024, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (16263, 'sm_103a', 152, 16, 2048, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16263, 'sm_107a', 208, 8, 1024, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16263, 'sm_107a', 216, 8, 1024, 'nsx_g2_ol2l', 'qe_mb10_tree', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16384, 'sm_100a', 148, 8, 1024, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (16384, 'sm_103a', 152, 16, 2048, 'nsx_g2_ol2l', 'st16_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16384, 'sm_107a', 208, 8, 1024, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16384, 'sm_107a', 216, 8, 1024, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16385, 'sm_100a', 148, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16385, 'sm_103a', 152, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16385, 'sm_107a', 208, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16385, 'sm_107a', 216, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16416, 'sm_100a', 148, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16416, 'sm_103a', 152, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16416, 'sm_107a', 208, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16416, 'sm_107a', 216, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16417, 'sm_100a', 148, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16417, 'sm_103a', 152, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16417, 'sm_107a', 208, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16417, 'sm_107a', 216, 8, 1032, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16576, 'sm_100a', 148, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_as_el_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411),
    (16576, 'sm_103a', 152, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16576, 'sm_107a', 208, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16576, 'sm_107a', 216, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16577, 'sm_100a', 148, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (16577, 'sm_103a', 152, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16577, 'sm_107a', 208, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16577, 'sm_107a', 216, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16640, 'sm_100a', 148, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (16640, 'sm_103a', 152, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16640, 'sm_107a', 208, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16640, 'sm_107a', 216, 8, 1040, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (16641, 'sm_100a', 148, 8, 1048, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (16641, 'sm_103a', 152, 8, 1048, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (16641, 'sm_107a', 208, 16, 2096, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (16641, 'sm_107a', 216, 8, 1048, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17024, 'sm_100a', 148, 8, 1064, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (17024, 'sm_103a', 152, 8, 1064, 'nsx_g2_ol2l', 'st8_lp2', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099),
    (17024, 'sm_107a', 208, 16, 2128, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17024, 'sm_107a', 216, 8, 1064, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17025, 'sm_100a', 148, 8, 1072, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (17025, 'sm_103a', 152, 8, 1072, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (17025, 'sm_107a', 208, 16, 2144, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17025, 'sm_107a', 216, 8, 1072, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17280, 'sm_100a', 148, 8, 1080, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (17280, 'sm_103a', 152, 8, 1080, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (17280, 'sm_107a', 208, 16, 2160, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17280, 'sm_107a', 216, 8, 1080, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 40, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17281, 'sm_100a', 148, 8, 1088, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (17281, 'sm_103a', 152, 8, 1088, 'nsx_g2_ol2l', 'st8_lp2', 64, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (17281, 'sm_107a', 208, 16, 2176, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (17281, 'sm_107a', 216, 16, 2176, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (19968, 'sm_100a', 148, 8, 1248, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (19968, 'sm_103a', 152, 8, 1248, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (19968, 'sm_107a', 208, 8, 1248, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (19968, 'sm_107a', 216, 8, 1248, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (19969, 'sm_100a', 148, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (19969, 'sm_103a', 152, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (19969, 'sm_107a', 208, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (19969, 'sm_107a', 216, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (20001, 'sm_100a', 148, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (20001, 'sm_103a', 152, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (20001, 'sm_107a', 208, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (20001, 'sm_107a', 216, 8, 1256, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (20736, 'sm_100a', 148, 8, 1296, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (20736, 'sm_103a', 152, 8, 1296, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (20736, 'sm_107a', 208, 8, 1296, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (20736, 'sm_107a', 216, 8, 1296, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 48, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (20737, 'sm_100a', 148, 8, 1304, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (20737, 'sm_103a', 152, 8, 1304, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (20737, 'sm_107a', 208, 8, 1304, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (20737, 'sm_107a', 216, 8, 1304, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (21632, 'sm_100a', 148, 8, 1352, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (21632, 'sm_103a', 152, 8, 1352, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (21632, 'sm_107a', 208, 8, 1352, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (21632, 'sm_107a', 216, 8, 1352, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (21633, 'sm_100a', 148, 8, 1360, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (21633, 'sm_103a', 152, 8, 1360, 'nsx_g2_ol2l', 'st8_lp2', 72, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (21633, 'sm_107a', 208, 8, 1360, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (21633, 'sm_107a', 216, 8, 1360, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (22464, 'sm_100a', 148, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (22464, 'sm_103a', 152, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (22464, 'sm_107a', 208, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (22464, 'sm_107a', 216, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (22465, 'sm_100a', 148, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (22465, 'sm_103a', 152, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (22465, 'sm_107a', 208, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (22465, 'sm_107a', 216, 8, 1408, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (23296, 'sm_100a', 148, 8, 1456, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (23296, 'sm_103a', 152, 8, 1456, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (23296, 'sm_107a', 208, 8, 1456, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (23296, 'sm_107a', 216, 8, 1456, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (23297, 'sm_100a', 148, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (23297, 'sm_103a', 152, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (23297, 'sm_107a', 208, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 64, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (23297, 'sm_107a', 216, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (23328, 'sm_100a', 148, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (23328, 'sm_103a', 152, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (23328, 'sm_107a', 208, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 64, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (23328, 'sm_107a', 216, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (23329, 'sm_100a', 148, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (23329, 'sm_103a', 152, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (23329, 'sm_107a', 208, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 64, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (23329, 'sm_107a', 216, 8, 1464, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411),
    (24192, 'sm_100a', 148, 8, 1512, 'nsx_g2_ol2l', 'st8_lp2', 88, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (24192, 'sm_103a', 152, 8, 1512, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (24192, 'sm_107a', 208, 8, 1512, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 64, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (24192, 'sm_107a', 216, 8, 1512, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 56, 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411),
    (24193, 'sm_100a', 148, 8, 1520, 'nsx_g2_ol2l', 'st8_lp2', 88, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (24193, 'sm_103a', 152, 8, 1520, 'nsx_g2_ol2l', 'st8_lp2', 80, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (24193, 'sm_107a', 208, 8, 1520, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 64, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (24193, 'sm_107a', 216, 8, 1520, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 64, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (32767, 'sm_100a', 148, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2', 112, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (32767, 'sm_103a', 152, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2', 112, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (32767, 'sm_107a', 208, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 80, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (32767, 'sm_107a', 216, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 80, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (32768, 'sm_100a', 148, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2', 112, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (32768, 'sm_103a', 152, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2', 112, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (32768, 'sm_107a', 208, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 80, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (32768, 'sm_107a', 216, 8, 2048, 'nsx_g2_ol2l', 'st8_lp2_tree_asel_qd', 80, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (32769, 'sm_100a', 148, 8, 2056, 'nsx_g2_ol2l', 'lp2_w8_adj', 112, 'dot_ao8', 'xrsg', 'wpu4_spred', 411),
    (32769, 'sm_103a', 152, 8, 2056, 'nsx_g2_ol2l', 'lp2_w8_adj', 112, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (32769, 'sm_107a', 208, 8, 2056, 'ss_st4', 'lp2_w8_adj_qe', 80, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (32769, 'sm_107a', 216, 8, 2056, 'ss_st4', 'lp2_w8_adj_qe', 80, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
    (65536, 'sm_100a', 148, 8, 4096, 'nsx_g2_ol2l', 'lp2_w8_adj', 224, 'dot_ao8', 'xrsg', 'wpu4_fadd', 411),
    (65536, 'sm_103a', 152, 8, 4096, 'nsx_g2_ol2l', 'lp2_w8_adj', 216, 'dot_ao8', 'xrsg', 'pt32', 3099),
    (65536, 'sm_107a', 208, 8, 4096, 'ss_st4', 'lp2_w8_adj_qe', 160, 'dot_ao8_l2l', 'xrsg', 'wpu4', 411),
    (65536, 'sm_107a', 216, 8, 4096, 'ss_st4', 'lp2_w8_adj_qe', 152, 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411),
)  # fmt: skip
RULES_COLUMNS = (
    "T",
    "sm_count",
    "arch",
    "T_pad",
    "launch",
    "P_K",
    "k1_units",
    "k1_grid",
    "P",
    "k7_units",
    "k7_grid",
    "rows_per_chunk",
    "G6",
    "R6",
    "k2_grid",
    "row_grid4",
    "k3_grid",
    "k4_grid",
    "k5_grid",
    "fwd_workspace_bytes",
    "bwd_workspace_bytes",
    "fn_split_program",
    "fn_split_program_t",
    "fn_split_grid",
    "fn_split_grid_t",
    "fwd_coef_program",
    "fwd_map_program",
    "bwd_token_program",
    "bwd_slab_program",
    "bwd_reduce_program",
    "bwd_reduce_grid",
    "k5_program",
    "fwd_finish_program",
)
RULES_ROWS = (
    (0, 148, 'sm_100a', 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'b16', 'tv128', 256, 256, '', '', '', 'xrsg', '', 0, '', 'production'),
    (0, 152, 'sm_103a', 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'b16', 'tv128', 256, 256, '', '', '', 'xrsg', '', 0, '', 'production'),
    (0, 208, 'sm_107a', 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'b16', 'tv128_t128', 256, 512, '', '', '', 'xrsg', '', 0, '', 'r2'),
    (0, 216, 'sm_107a', 0, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'b16', 'tv128_t128', 256, 512, '', '', '', 'xrsg', '', 0, '', 'r2'),
    (1, 148, 'sm_100a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 1, 2, 1, 1, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (1, 152, 'sm_103a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 1, 2, 1, 1, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (1, 208, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 1, 2, 1, 1, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (1, 216, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 1, 2, 1, 1, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (2, 148, 'sm_100a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 2, 4, 1, 1, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (2, 152, 'sm_103a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 2, 4, 1, 2, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (2, 208, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 2, 4, 2, 2, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (2, 216, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 2, 4, 2, 2, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (3, 148, 'sm_100a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 3, 6, 2, 2, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (3, 152, 'sm_103a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 3, 6, 2, 3, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (3, 208, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 3, 6, 3, 3, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (3, 216, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 3, 6, 3, 3, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (7, 148, 'sm_100a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 7, 14, 4, 4, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (7, 152, 'sm_103a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 7, 14, 4, 7, 3555584, 4755968, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (7, 208, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 7, 14, 7, 7, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (7, 216, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 1, 8, 1, 7, 14, 7, 7, 3555584, 4755968, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (63, 148, 'sm_100a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 63, 126, 32, 32, 3555584, 4756736, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (63, 152, 'sm_103a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 63, 126, 32, 63, 3555584, 4756736, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (63, 208, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 63, 126, 63, 63, 3555584, 4756736, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (63, 216, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 63, 126, 63, 63, 3555584, 4756736, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (64, 148, 'sm_100a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 64, 128, 32, 32, 3555584, 4756736, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (64, 152, 'sm_103a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 64, 128, 32, 64, 3555584, 4756736, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (64, 208, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 64, 128, 64, 64, 3555584, 4756736, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (64, 216, 'sm_107a', 128, True, 32, 32, 32, 1, 64, 64, 64, 8, 8, 4, 64, 128, 64, 64, 3555584, 4756736, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (65, 148, 'sm_100a', 128, True, 32, 32, 32, 2, 128, 128, 64, 9, 8, 5, 65, 130, 33, 33, 3555584, 6329856, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (65, 152, 'sm_103a', 128, True, 32, 32, 32, 2, 128, 128, 64, 9, 8, 5, 65, 130, 33, 65, 3555584, 6329856, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (65, 208, 'sm_107a', 128, True, 32, 32, 32, 2, 128, 128, 64, 9, 8, 5, 65, 130, 65, 65, 3555584, 6329856, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (65, 216, 'sm_107a', 128, True, 32, 32, 32, 2, 128, 128, 64, 9, 8, 5, 65, 130, 65, 65, 3555584, 6329856, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (127, 148, 'sm_100a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 127, 254, 64, 64, 3555584, 6330624, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (127, 152, 'sm_103a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 127, 254, 64, 127, 3555584, 6330624, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (127, 208, 'sm_107a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 127, 254, 127, 127, 3555584, 6330624, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (127, 216, 'sm_107a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 127, 254, 127, 127, 3555584, 6330624, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (128, 148, 'sm_100a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 128, 256, 64, 64, 3555584, 6330624, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (128, 152, 'sm_103a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 128, 256, 64, 128, 3555584, 6330624, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (128, 208, 'sm_107a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 128, 256, 128, 128, 3555584, 6330624, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (128, 216, 'sm_107a', 128, True, 32, 32, 32, 2, 128, 128, 64, 16, 8, 8, 128, 256, 128, 128, 3555584, 6330624, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (129, 148, 'sm_100a', 256, True, 32, 64, 64, 2, 128, 128, 128, 17, 8, 9, 129, 258, 65, 65, 3965184, 6367744, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (129, 152, 'sm_103a', 256, True, 32, 64, 64, 2, 128, 128, 128, 17, 8, 9, 129, 258, 65, 129, 3965184, 6367744, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (129, 208, 'sm_107a', 256, True, 32, 64, 64, 3, 192, 192, 64, 17, 8, 9, 129, 258, 129, 129, 3965184, 7940608, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (129, 216, 'sm_107a', 256, True, 32, 64, 64, 3, 192, 192, 64, 17, 8, 9, 129, 258, 129, 129, 3965184, 7940608, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (255, 148, 'sm_100a', 256, True, 32, 64, 64, 2, 128, 128, 128, 32, 8, 16, 255, 510, 128, 128, 3965184, 6369536, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (255, 152, 'sm_103a', 256, True, 32, 64, 64, 2, 128, 128, 128, 32, 8, 16, 255, 510, 128, 255, 3965184, 6369536, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (255, 208, 'sm_107a', 256, True, 32, 64, 64, 2, 128, 128, 128, 32, 8, 16, 255, 510, 255, 255, 3965184, 6369536, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (255, 216, 'sm_107a', 256, True, 32, 64, 64, 2, 128, 128, 128, 32, 8, 16, 255, 510, 255, 255, 3965184, 6369536, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (1000, 148, 'sm_100a', 1024, True, 16, 128, 128, 2, 128, 128, 512, 125, 8, 63, 592, 2000, 500, 296, 4784384, 6602752, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (1000, 152, 'sm_103a', 1024, True, 16, 128, 128, 2, 128, 128, 512, 125, 8, 63, 608, 2000, 500, 1000, 4784384, 6602752, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (1000, 208, 'sm_107a', 1024, True, 16, 128, 128, 3, 192, 192, 384, 125, 8, 63, 832, 2000, 1000, 1000, 4784384, 8175616, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (1000, 216, 'sm_107a', 1024, True, 16, 128, 128, 3, 192, 192, 384, 125, 8, 63, 864, 2000, 1000, 1000, 4784384, 8175616, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (1024, 148, 'sm_100a', 1024, True, 16, 128, 128, 2, 128, 128, 512, 128, 8, 64, 592, 2048, 512, 296, 4784384, 6603008, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'xrsg', 's27_wcvt_fadd_fshfl', 411, 'production', 'production'),
    (1024, 152, 'sm_103a', 1024, True, 16, 128, 128, 2, 128, 128, 512, 128, 8, 64, 608, 2048, 512, 1024, 4784384, 6603008, 'b16', 'tv128', 256, 256, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (1024, 208, 'sm_107a', 1024, True, 16, 128, 128, 3, 192, 192, 384, 128, 8, 64, 832, 2048, 1024, 1024, 4784384, 8175872, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (1024, 216, 'sm_107a', 1024, True, 16, 128, 128, 3, 192, 192, 384, 128, 8, 64, 864, 2048, 1024, 1024, 4784384, 8175872, 'b16', 'tv128_t128', 256, 512, 'st16_w4_tree', 'nsx_st5_mb1', 'ao22_rae7_rf3_pd1_l2l_ntr', 'xrsg', 's27_wcvt_spred', 411, 't128', 'r2'),
    (2049, 148, 'sm_100a', 2176, True, 8, 136, 136, 11, 704, 148, 192, 257, 8, 129, 592, 4098, 1025, 2049, 4886784, 21107200, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_spred', 411, 't128', 'production'),
    (2049, 152, 'sm_103a', 2176, True, 8, 136, 136, 7, 448, 152, 320, 257, 8, 129, 608, 4098, 1025, 2049, 4886784, 14815744, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx_st5_mb1', 'ao22_rae7_pd1_ntr', 'dynr', 'pt64', 1563, 't128', 'production'),
    (2049, 208, 'sm_107a', 2176, True, 32, 544, 416, 3, 192, 192, 704, 257, 8, 129, 832, 4098, 2049, 2049, 10109184, 8524288, 'b16', 'tv128_t128', 256, 512, 'pfl2_qmt', 'nsx_g2_ol2l', 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411, 't128', 'r2'),
    (2049, 216, 'sm_107a', 2176, True, 32, 544, 432, 3, 192, 192, 704, 257, 8, 129, 864, 4098, 2049, 2049, 10109184, 8524288, 'b16', 'tv128_t128', 256, 512, 'pfl2_qmt', 'nsx_g2_ol2l', 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411, 't128', 'r2'),
    (4096, 148, 'sm_100a', 4096, True, 8, 256, 256, 16, 1024, 148, 256, 296, 16, 256, 592, 8192, 2048, 4096, 6422784, 29529344, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'ns96_g2_ol2l', 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4_fadd', 411, 't128', 'production'),
    (4096, 152, 'sm_103a', 4096, True, 8, 256, 256, 16, 1024, 152, 256, 304, 16, 256, 608, 8192, 2048, 4096, 6422784, 29530368, 'b16', 'tv128', 256, 256, 'qe_mb10_tree', 'nsx', 'ao22_rae7_pd1_ntr', 'dynr', 'pt32', 3099, 't128', 'production'),
    (4096, 208, 'sm_107a', 4096, True, 32, 1024, 416, 13, 832, 208, 320, 416, 16, 256, 832, 8192, 4096, 4096, 16253184, 24826112, 'b16', 'tv128_t128', 256, 512, 'pfl2_qmt', 'nsx_g2_ol2l', 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411, 't128', 'r2'),
    (4096, 216, 'sm_107a', 4096, True, 32, 1024, 432, 13, 832, 216, 320, 432, 16, 256, 864, 8192, 4096, 4096, 16253184, 24828160, 'b16', 'tv128_t128', 256, 512, 'pfl2_qmt', 'nsx_g2_ol2l', 'ao22_rae7_pd1_tma_ntr', 'xrsg', 'wpu4', 411, 't128', 'r2'),
    (8192, 148, 'sm_100a', 8192, True, 16, 1024, 296, 16, 1024, 148, 512, 296, 32, 512, 592, 16384, 4096, 8192, 16253184, 30708992, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'ns96_g2_ol2l', 'ao22_rae7_pd1_tma_ntr', 'dynr', 'wpu4_fadd', 411, 't128', 'production'),
    (8192, 152, 'sm_103a', 8192, True, 16, 1024, 304, 7, 448, 152, 1216, 304, 32, 512, 608, 16384, 4096, 8192, 16253184, 16554240, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao22_rae7_pd1_ntr', 'xrsg', 'pt32', 3099, 't128', 'production'),
    (8192, 208, 'sm_107a', 8192, True, 16, 1024, 416, 13, 832, 208, 640, 416, 24, 512, 832, 16384, 8192, 8192, 16253184, 26005760, 'b16', 'tv128_t128', 256, 512, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411, 't128', 'r2'),
    (8192, 216, 'sm_107a', 8192, True, 16, 1024, 432, 10, 640, 216, 832, 432, 24, 512, 864, 16384, 8192, 8192, 16253184, 21289216, 'b16', 'tv128_t128', 256, 512, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao22_rae7_pd1_ntr', 'xrsg', 'wpu4', 411, 't128', 'r2'),
    (14928, 148, 'sm_100a', 14976, True, 8, 936, 296, 9, 576, 148, 1664, 296, 56, 933, 592, 29856, 7464, 7464, 15126784, 21652736, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (14928, 152, 'sm_103a', 14976, True, 16, 1872, 304, 7, 448, 152, 2176, 304, 56, 933, 608, 29856, 7464, 7464, 27107584, 18508032, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (14928, 208, 'sm_107a', 14976, True, 16, 1872, 416, 13, 832, 208, 1152, 416, 40, 933, 832, 29856, 14928, 7464, 27107584, 27959552, 'b16', 'tv128_t128', 256, 512, 'g2_el', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (14928, 216, 'sm_107a', 14976, True, 16, 1872, 432, 10, 640, 216, 1536, 432, 40, 933, 864, 29856, 14928, 7464, 27107584, 23243008, 'b16', 'tv128_t128', 256, 512, 'g2_el', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (15288, 148, 'sm_100a', 15360, True, 8, 960, 296, 16, 1024, 148, 960, 296, 56, 956, 592, 30576, 7644, 7644, 15433984, 32773376, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411, 'production', 'production'),
    (15288, 152, 'sm_103a', 15360, True, 16, 1920, 304, 7, 448, 152, 2240, 304, 56, 956, 608, 30576, 7644, 7644, 27721984, 18618624, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (15288, 208, 'sm_107a', 15360, True, 8, 960, 416, 16, 1024, 208, 960, 416, 40, 956, 832, 30576, 15288, 7644, 15433984, 32788736, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411, 'production', 'r2'),
    (15288, 216, 'sm_107a', 15360, True, 16, 1920, 432, 10, 640, 216, 1536, 432, 40, 956, 864, 30576, 15288, 7644, 27721984, 23353600, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (15566, 148, 'sm_100a', 15616, True, 8, 976, 296, 9, 576, 148, 1792, 296, 56, 973, 592, 31132, 7783, 7783, 15638784, 21837056, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (15566, 152, 'sm_103a', 15616, True, 16, 1952, 304, 7, 448, 152, 2240, 304, 56, 973, 608, 31132, 7783, 7783, 28131584, 18692352, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (15566, 208, 'sm_107a', 15616, True, 8, 976, 416, 13, 832, 208, 1216, 416, 40, 973, 832, 31132, 15566, 7783, 15638784, 28143872, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (15566, 216, 'sm_107a', 15616, True, 8, 976, 432, 10, 640, 216, 1600, 432, 40, 973, 864, 31132, 15566, 7783, 15638784, 23427328, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (15590, 148, 'sm_100a', 15616, True, 8, 976, 296, 9, 576, 148, 1792, 296, 56, 975, 592, 31180, 7795, 7795, 15638784, 21837056, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (15590, 152, 'sm_103a', 15616, True, 16, 1952, 304, 7, 448, 152, 2240, 304, 56, 975, 608, 31180, 7795, 7795, 28131584, 18692352, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (15590, 208, 'sm_107a', 15616, True, 8, 976, 416, 13, 832, 208, 1216, 416, 40, 975, 832, 31180, 15590, 7795, 15638784, 28143872, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (15590, 216, 'sm_107a', 15616, True, 8, 976, 432, 10, 640, 216, 1600, 432, 40, 975, 864, 31180, 15590, 7795, 15638784, 23427328, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16114, 148, 'sm_100a', 16128, True, 8, 1008, 296, 9, 576, 148, 1792, 296, 56, 1008, 592, 32228, 8057, 8057, 16048384, 21984512, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (16114, 152, 'sm_103a', 16128, True, 16, 2016, 304, 7, 448, 152, 2304, 304, 56, 1008, 608, 32228, 8057, 8057, 28950784, 18839808, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (16114, 208, 'sm_107a', 16128, True, 8, 1008, 416, 13, 832, 208, 1280, 416, 40, 1008, 832, 32228, 16114, 8057, 16048384, 28291328, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16114, 216, 'sm_107a', 16128, True, 8, 1008, 432, 10, 640, 216, 1664, 432, 40, 1008, 864, 32228, 16114, 8057, 16048384, 23574784, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16132, 148, 'sm_100a', 16256, True, 8, 1016, 296, 23, 1472, 148, 704, 296, 56, 1009, 592, 32264, 8066, 8066, 16150784, 44041472, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (16132, 152, 'sm_103a', 16256, True, 16, 2032, 304, 7, 448, 152, 2368, 304, 56, 1009, 608, 32264, 8066, 8066, 29155584, 18876672, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (16132, 208, 'sm_107a', 16256, True, 8, 1016, 416, 13, 832, 208, 1280, 416, 40, 1009, 832, 32264, 16132, 8066, 16150784, 28328192, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16132, 216, 'sm_107a', 16256, True, 8, 1016, 432, 10, 640, 216, 1664, 432, 40, 1009, 864, 32264, 16132, 8066, 16150784, 23611648, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16164, 148, 'sm_100a', 16256, True, 8, 1016, 296, 23, 1472, 148, 704, 296, 56, 1011, 592, 32328, 8082, 8082, 16150784, 44041472, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (16164, 152, 'sm_103a', 16256, True, 16, 2032, 304, 7, 448, 152, 2368, 304, 56, 1011, 608, 32328, 8082, 8082, 29155584, 18876672, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (16164, 208, 'sm_107a', 16256, True, 8, 1016, 416, 13, 832, 208, 1280, 416, 40, 1011, 832, 32328, 16164, 8082, 16150784, 28328192, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16164, 216, 'sm_107a', 16256, True, 8, 1016, 432, 10, 640, 216, 1664, 432, 40, 1011, 864, 32328, 16164, 8082, 16150784, 23611648, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16263, 148, 'sm_100a', 16384, True, 8, 1024, 296, 16, 1024, 148, 1024, 296, 56, 1017, 592, 32526, 8132, 8132, 16253184, 33068288, 'b16', 'tv128', 256, 256, 'st8_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4_fadd', 411, 'production', 'production'),
    (16263, 152, 'sm_103a', 16384, True, 16, 2048, 304, 7, 448, 152, 2368, 304, 56, 1017, 608, 32526, 8132, 8132, 29360384, 18913536, 'b16', 'tv128', 256, 256, 'st16_lp2_as_el_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (16263, 208, 'sm_107a', 16384, True, 8, 1024, 416, 13, 832, 208, 1280, 416, 40, 1017, 832, 32526, 16263, 8132, 16253184, 28365056, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (16263, 216, 'sm_107a', 16384, True, 8, 1024, 432, 10, 640, 216, 1664, 432, 40, 1017, 864, 32526, 16263, 8132, 16253184, 23648512, 'b16', 'tv128_t128', 256, 512, 'qe_mb10_tree', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (20001, 148, 'sm_100a', 20096, True, 8, 1256, 296, 9, 576, 148, 2240, 296, 72, 1251, 592, 40002, 10001, 10001, 19222784, 23127296, 'b16', 'tv128', 256, 256, 'st8_lp2', 'nsx_g2_ol2l', 'dot_ao8', 'xrsg', 'wpu4_spred', 411, 'production', 'production'),
    (20001, 152, 'sm_103a', 20096, True, 8, 1256, 304, 7, 448, 152, 2880, 304, 72, 1251, 608, 40002, 10001, 10001, 19222784, 19982592, 'b16', 'tv128', 256, 256, 'st8_lp2', 'nsx_g2_ol2l', 'dot_ao8', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (20001, 208, 'sm_107a', 20096, True, 8, 1256, 416, 13, 832, 208, 1600, 416, 56, 1251, 832, 40002, 20001, 10001, 19222784, 29434112, 'b16', 'tv128_t128', 256, 512, 'st8_lp2_tree_asel_qd', 'nsx_g2_ol2l', 'ao15_rae8_tma_ntr', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (20001, 216, 'sm_107a', 20096, True, 8, 1256, 432, 10, 640, 216, 2048, 432, 48, 1251, 864, 40002, 20001, 10001, 19222784, 24717568, 'b16', 'tv128_t128', 256, 512, 'st8_lp2_tree_asel_qd', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (32768, 148, 'sm_100a', 32768, True, 8, 2048, 296, 16, 1024, 148, 2048, 296, 112, 2048, 592, 65536, 16384, 16384, 29360384, 37786880, 'b16', 'tv128', 256, 256, 'st8_lp2', 'nsx_g2_ol2l', 'dot_ao8', 'xrsg', 'wpu4_fadd', 411, 'production', 'production'),
    (32768, 152, 'sm_103a', 32768, True, 8, 2048, 304, 19, 1216, 152, 1728, 304, 112, 2048, 608, 65536, 16384, 16384, 29360384, 42506496, 'b16', 'tv128', 256, 256, 'st8_lp2', 'nsx_g2_ol2l', 'dot_ao8', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (32768, 208, 'sm_107a', 32768, True, 8, 2048, 416, 13, 832, 208, 2560, 416, 80, 2048, 832, 65536, 32768, 16384, 29360384, 33083648, 'b16', 'tv128_t128', 256, 512, 'st8_lp2_tree_asel_qd', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (32768, 216, 'sm_107a', 32768, True, 8, 2048, 432, 27, 1728, 216, 1216, 432, 80, 2048, 864, 65536, 32768, 16384, 29360384, 55105792, 'b16', 'tv128_t128', 256, 512, 'st8_lp2_tree_asel_qd', 'nsx_g2_ol2l', 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411, 'production', 'r2'),
    (65536, 148, 'sm_100a', 65536, True, 8, 4096, 296, 16, 1024, 148, 4096, 296, 224, 4096, 592, 131072, 32768, 32768, 55574784, 47224064, 'b16', 'tv128', 256, 256, 'lp2_w8_adj', 'nsx_g2_ol2l', 'dot_ao8', 'xrsg', 'wpu4_fadd', 411, 'production', 'production'),
    (65536, 152, 'sm_103a', 65536, True, 8, 4096, 304, 19, 1216, 152, 3456, 304, 216, 4096, 608, 131072, 32768, 32768, 55574784, 51943680, 'b16', 'tv128', 256, 256, 'lp2_w8_adj', 'nsx_g2_ol2l', 'dot_ao8', 'xrsg', 'pt32', 3099, 'production', 'production'),
    (65536, 208, 'sm_107a', 65536, True, 8, 4096, 416, 13, 832, 208, 5056, 416, 160, 4096, 832, 131072, 65536, 32768, 55574784, 42520832, 'b16', 'tv128_t128', 256, 512, 'lp2_w8_adj_qe', 'ss_st4', 'dot_ao8_l2l', 'xrsg', 'wpu4', 411, 'production', 'r2'),
    (65536, 216, 'sm_107a', 65536, True, 8, 4096, 432, 27, 1728, 216, 2432, 432, 152, 4096, 864, 131072, 65536, 32768, 55574784, 64542976, 'b16', 'tv128_t128', 256, 512, 'lp2_w8_adj_qe', 'ss_st4', 'dot_ao8_l2l', 'xrsg', 'wpu4_fadd', 411, 'production', 'r2'),
)  # fmt: skip
LAUNCH_FACTS = {
    "cake_mhc_train_fn_split": ((256, 1, 1), 0, 1),
    "cake_mhc_train_fn_split__tv128__t": ((256, 1, 1), 0, 1),
    "cake_mhc_train_fn_split__tv128_t128__t": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_map__nsx_st5_mb1": ((192, 1, 1), 144384, 1),
    "cake_mhc_train_fwd_map__nsx": ((192, 1, 1), 115712, 2),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l": ((192, 1, 1), 115712, 2),
    "cake_mhc_train_fwd_map__nsx_st5_mb1__a3": ((192, 1, 1), 144384, 1),
    "cake_mhc_train_fwd_map__ns96_g2_ol2l": ((192, 1, 1), 115712, 2),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__a3": ((192, 1, 1), 115712, 2),
    "cake_mhc_train_fwd_map__nsx_st5_mb1__a7": ((192, 1, 1), 144384, 1),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__a7": ((192, 1, 1), 115712, 2),
    "cake_mhc_train_fwd_map__hi": ((192, 1, 1), 115712, 2),
    "cake_mhc_train_fwd_coef__qe_mb10_tree": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_coef__rt64": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_coef__st16_w4_tree": ((128, 1, 1), 6400, 1),
    "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd": ((128, 1, 1), 12800, 1),
    "cake_mhc_train_fwd_coef__qe_mb10": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_coef__st8_lp2": ((128, 1, 1), 6400, 1),
    "cake_mhc_train_fwd_coef__qe_mb10_pu4": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_coef__lp2_w8_adj": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd": ((128, 1, 1), 6400, 1),
    "cake_mhc_train_fwd_coef__pfl2_qmt": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_coef__g2_el": ((128, 1, 1), 25600, 1),
    "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd": ((128, 1, 1), 6400, 1),
    "cake_mhc_train_fwd_coef__lp2_w8_adj_qe": ((128, 1, 1), 0, 1),
    "cake_mhc_train_fwd_collapse": ((256, 1, 1), 0, 1),
    "cake_mhc_train_fwd_finish": ((256, 1, 1), 0, 1),
    "cake_mhc_train_fwd_finish__r2": ((256, 1, 1), 0, 1),
    "cake_mhc_train_bwd_prepare__g2sm_le1024": ((256, 1, 1), 1280, 1),
    "cake_mhc_train_bwd_prepare__t128": ((128, 1, 1), 640, 1),
    "cake_mhc_train_bwd_prepare__t128__a3": ((128, 1, 1), 640, 1),
    "cake_mhc_train_bwd_prepare__gt8192": ((256, 1, 1), 1280, 1),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__dot_ao8": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a3": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__rt64": ((256, 1, 1), 71296, 2),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__dot_ao8__a3": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a3": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a7": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__dot_ao8_l2l": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a7": ((256, 1, 1), 91136, 2),
    "cake_mhc_train_bwd_slab__dynr": ((384, 1, 1), 228480, 1),
    "cake_mhc_train_bwd_slab__xrsg": ((384, 1, 1), 232448, 1),
    "cake_mhc_train_bwd_slab__xrsg__a7": ((384, 1, 1), 232448, 1),
    "cake_mhc_train_bwd_reduce__pt64": ((64, 1, 1), 2688, 1),
    "cake_mhc_train_bwd_reduce__pt32": ((32, 1, 1), 2432, 1),
    "cake_mhc_train_bwd_reduce__s27_wcvt_fadd_fshfl": ((256, 1, 1), 4224, 1),
    "cake_mhc_train_bwd_reduce__wpu4_spred": ((256, 1, 1), 4224, 1),
    "cake_mhc_train_bwd_reduce__wpu4_fadd": ((256, 1, 1), 4224, 1),
    "cake_mhc_train_bwd_reduce__s27_wcvt_spred": ((256, 1, 1), 4224, 1),
    "cake_mhc_train_bwd_reduce__wpu4": ((256, 1, 1), 4224, 1),
}
RECORD_ARCHES = {
    "cake_mhc_train_fn_split": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fn_split__tv128__t": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fn_split__tv128_t128__t": ("sm_107a",),
    "cake_mhc_train_fwd_map__nsx_st5_mb1": ("sm_103a",),
    "cake_mhc_train_fwd_map__nsx": ("sm_103a",),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l": ("sm_103a",),
    "cake_mhc_train_fwd_map__nsx_st5_mb1__a3": ("sm_100a",),
    "cake_mhc_train_fwd_map__ns96_g2_ol2l": ("sm_100a",),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__a3": ("sm_100a",),
    "cake_mhc_train_fwd_map__nsx_st5_mb1__a7": ("sm_107a",),
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__a7": ("sm_107a",),
    "cake_mhc_train_fwd_map__hi": ("sm_107a",),
    "cake_mhc_train_fwd_coef__qe_mb10_tree": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__st16_w4_tree": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_coef__qe_mb10": ("sm_103a",),
    "cake_mhc_train_fwd_coef__st8_lp2": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_coef__qe_mb10_pu4": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_coef__lp2_w8_adj": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd": ("sm_100a",),
    "cake_mhc_train_fwd_coef__pfl2_qmt": ("sm_107a",),
    "cake_mhc_train_fwd_coef__g2_el": ("sm_107a",),
    "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd": ("sm_107a",),
    "cake_mhc_train_fwd_coef__lp2_w8_adj_qe": ("sm_107a",),
    "cake_mhc_train_fwd_coef__rt64": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_collapse": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_fwd_finish": ("sm_100a", "sm_103a"),
    "cake_mhc_train_fwd_finish__r2": ("sm_107a",),
    "cake_mhc_train_bwd_prepare__g2sm_le1024": ("sm_100a",),
    "cake_mhc_train_bwd_prepare__t128": ("sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_prepare__t128__a3": ("sm_100a",),
    "cake_mhc_train_bwd_prepare__gt8192": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr": ("sm_103a",),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr": ("sm_103a",),
    "cake_mhc_train_bwd_token__dot_ao8": ("sm_103a",),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a3": ("sm_100a",),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr": ("sm_100a", "sm_107a"),
    "cake_mhc_train_bwd_token__dot_ao8__a3": ("sm_100a",),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a3": ("sm_100a",),
    "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr": ("sm_107a",),
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a7": ("sm_107a",),
    "cake_mhc_train_bwd_token__dot_ao8_l2l": ("sm_107a",),
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a7": ("sm_107a",),
    "cake_mhc_train_bwd_token__rt64": ("sm_100a", "sm_103a", "sm_107a"),
    "cake_mhc_train_bwd_slab__dynr": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab__xrsg": ("sm_100a", "sm_103a"),
    "cake_mhc_train_bwd_slab__xrsg__a7": ("sm_107a",),
    "cake_mhc_train_bwd_reduce__pt64": ("sm_103a",),
    "cake_mhc_train_bwd_reduce__pt32": ("sm_103a",),
    "cake_mhc_train_bwd_reduce__s27_wcvt_fadd_fshfl": ("sm_100a",),
    "cake_mhc_train_bwd_reduce__wpu4_spred": ("sm_100a",),
    "cake_mhc_train_bwd_reduce__wpu4_fadd": ("sm_100a", "sm_107a"),
    "cake_mhc_train_bwd_reduce__s27_wcvt_spred": ("sm_107a",),
    "cake_mhc_train_bwd_reduce__wpu4": ("sm_107a",),
}
# The backward prepare's CTA forms of record -> library (generated from the records; the production record's library
# everywhere, the 128-thread one-token form's where k5_program routes it).
K5_FORM_LIBRARY = {
    "production": "cake_mhc_train_bwd_prepare",
    "t128": "cake_mhc_train_bwd_prepare__t128",
}
K5_FORM_OF_LIBRARY = {library: form for form, library in K5_FORM_LIBRARY.items()}
# The token-count bands (above, at_most] where the per-call PDL attribute is off, per architecture, edge and call path
# (generated from the table of record; empty where every edge keeps the attribute).
PDL_OFF_BANDS = {
    "sm_100a": {
        "K1->K3": {
            "forward": ((4096, 8192),),
        },
        "K1->K5": {
            "checkpoint": ((4096, 8192),),
        },
        "K6->K7": {
            "checkpoint": ((1024, 8192),),
        },
        "K7->K8": {
            "backward": ((1024, 8192),),
            "checkpoint": ((1024, 4096),),
        },
    },
    "sm_103a": {
        "K1->K3": {
            "forward": ((4096, 16263),),
        },
        "K1->K5": {
            "checkpoint": ((4096, 16263),),
        },
        "K6->K7": {
            "checkpoint": ((4096, 8192),),
        },
        "K7->K8": {
            "backward": ((0, 4096),),
            "checkpoint": ((0, 4096),),
        },
    },
    "sm_107a": {},
}
EVERYWHERE = {
    "cake_mhc_train_fwd_collapse",
    "cake_mhc_train_bwd_prepare",
}
# The residual mixing's record per architecture (when the master defines FWD_FINISH_PROGRAM; the production
# record everywhere otherwise).
FINISH_OF = {
    arch: FINISH_LIBRARY[TABLES.get("FWD_FINISH_PROGRAM", {}).get(arch, "production")]
    for arch in ARCHES
}
LIBRARIES = sorted(LIBRARY_ARCHES)
# The kernels whose records a `program` selection rule may select, with their library table and host rule (K1 on
# every architecture; K6 on sm_107a; K7 on sm_100a and sm_103a; K8 on sm_100a and sm_107a).
PROGRAM_RULE_KERNELS = {
    "cake_mhc_train_fwd_coef": (COEF_LIBRARY, "fwd_coef_program"),
    "cake_mhc_train_bwd_token": (TOKEN_LIBRARY, "bwd_token_program"),
    "cake_mhc_train_bwd_slab": (SLAB_LIBRARY, "bwd_slab_program"),
    "cake_mhc_train_bwd_reduce": (REDUCE_LIBRARY, "bwd_reduce_program"),
}


def _rows() -> list[dict]:
    return [dict(zip(RULES_COLUMNS, row, strict=False)) for row in RULES_ROWS]


def _first_programs(fn, arch: str, sms) -> list:
    """The programs ``fn(T, sm, arch)`` routes on the architecture's boards, by the smallest T routing each -- the
    generator's record order of the program-rule records: the fine sweep 1..70000 step 97 and DISPATCH_T on the
    boards, then DISPATCH_T on every SM count of record (ties by name)."""
    first: dict = {}
    for T in sorted(set(DISPATCH_T) | set(range(1, 70001, 97))):
        for sm in sms:
            first.setdefault(fn(T, sm, arch), T)
    for sm in SM_COUNTS:
        for T in DISPATCH_T:
            first.setdefault(fn(T, sm, arch), T)
    return sorted(first, key=lambda p: (first[p], p))


def _is(name: str, *prefixes: str) -> bool:
    return loader.MODULES[name]["library"].startswith(prefixes)


@pytest.mark.cpu_only
def test_rules_table_is_the_harness_table_of_record() -> None:
    expected = _rows()
    assert len(expected) == 28 * len(BOARDS)
    assert loader.RULES_SM_COUNTS == SM_COUNTS
    assert tuple(tuple(board) for board in loader.RULES_BOARDS) == BOARDS
    assert sorted({row["T"] for row in expected}) == sorted(loader.RULES_T_VALUES)
    assert {(row["arch"], row["sm_count"]) for row in expected} == set(BOARDS)
    actual = loader.rules_table()
    assert len(actual) == len(expected)
    for want, got in zip(expected, actual, strict=False):
        assert set(got) == set(RULES_COLUMNS), sorted(set(got) ^ set(RULES_COLUMNS))
        assert got == want, {
            key: (want[key], got.get(key)) for key in want if want[key] != got.get(key)
        }
    for want in expected:
        assert loader.rules_row(want["T"], want["sm_count"], want["arch"]) == want


@pytest.mark.cpu_only
def test_host_plan_matches_the_table() -> None:
    for row in _rows():
        if row["T"] == 0:
            continue
        plan = loader.host_plan(row["T"], row["sm_count"], row["arch"])
        assert plan["arch"] == row["arch"] and plan["sm_count"] == row["sm_count"]
        assert plan["t_pad"] == row["T_pad"] == loader.t_pad(row["T"])
        assert plan["p_k"] == row["P_K"] and 1 << plan["log2_pk"] == row["P_K"]
        assert plan["k1_units"] == row["k1_units"] and plan["k1_grid"] == row["k1_grid"]
        assert plan["k2_grid"] == row["k2_grid"]
        assert plan["parts"] == row["P"] and plan["k7_units"] == row["k7_units"]
        assert (
            plan["k7_grid"] == row["k7_grid"]
            and plan["rows_per_chunk"] == row["rows_per_chunk"]
        )
        assert plan["chunks"] == row["G6"] and plan["rows_per_cta"] == row["R6"]
        assert (
            plan["k3_grid"]
            == row["k3_grid"]
            == loader.k3_grid(row["T"])
            == loader.row_cta_grid(row["T"], 1, 2)
        )
        assert (
            plan["k4_grid"]
            == loader.k4_grid(row["T"])
            == loader.row_cta_grid(row["T"], 4, 2)
        )  # production
        # the rules table's k4_grid = the launch grid of the residual mixing's program of record (T under r2 on
        # sm_107a, ceil(T / 2) under production)
        assert row["k4_grid"] == loader.k4_grid_of_record(row["T"], row["arch"])
        assert (
            row["fwd_finish_program"]
            == loader.fwd_finish_program(row["arch"])
            == loader.FWD_FINISH_PROGRAM[row["arch"]]
        )
        assert row["k4_grid"] == loader.row_cta_grid(
            row["T"], loader.FWD_FINISH_ROWS[row["fwd_finish_program"]], 2
        )
        # the backward prepare: the form of record (K5_FORM_BANDS: the 128-thread one-token form inside its band, the
        # two-token production form elsewhere) on its one-trip grid, which the persistent grid_per_sm x SMs grid caps
        # inside a production band (sm_103a T <= 4096; sm_100a T <= 1024)
        form = loader.k5_program(row["T"], row["sm_count"], row["arch"])
        assert form == row["k5_program"] and form in loader.K5_FORM_ROWS, (
            row["T"],
            row["arch"],
            form,
        )
        one_trip = loader.row_cta_grid(row["T"], loader.K5_FORM_ROWS[form], 1)
        caps = [
            g
            for a, b, g in loader.K5_GRID_BANDS[row["arch"]]
            if a < row["T"] and (b is None or row["T"] <= b)
        ]
        want = (
            min(one_trip, caps[0] * row["sm_count"])
            if caps and form == "production"
            else one_trip
        )
        assert (
            plan["k5_grid"]
            == row["k5_grid"]
            == loader.k5_grid(row["T"], row["sm_count"], row["arch"])
            == want
        )
        assert plan["m_tiles"] == -(-row["T"] // loader.TILE_T)
        assert plan["rows_per_chunk"] * plan["parts"] >= row["T"]
        assert plan["chunks"] * plan["rows_per_cta"] >= row["T"]


@pytest.mark.cpu_only
def test_program_columns_follow_the_rules_of_record() -> None:
    for row in _rows():
        T, sm, arch = row["T"], row["sm_count"], row["arch"]
        if T == 0:
            assert (
                row["fwd_map_program"]
                == row["fwd_coef_program"]
                == row["bwd_token_program"]
                == ""
            )
        else:
            assert row["fwd_map_program"] == loader.fwd_map_program(T, sm, arch)
            assert row["fwd_coef_program"] == loader.fwd_coef_program(T, sm, arch)
            assert row["bwd_token_program"] == loader.bwd_token_program(T, sm, arch)
            assert row["fwd_map_program"] in MAP_LIBRARY
            assert row["fwd_coef_program"] in COEF_LIBRARY
            assert row["bwd_token_program"] in TOKEN_LIBRARY
        assert (
            row["fn_split_program"]
            == loader.fn_split_program(arch, 0)
            == loader.FN_SPLIT_PROGRAM[arch][0]
        )
        assert (
            row["fn_split_program_t"]
            == loader.fn_split_program(arch, 1)
            == loader.FN_SPLIT_PROGRAM[arch][1]
        )
        assert (
            row["fn_split_grid"]
            == loader.fn_split_grid(arch, 0)
            == loader.FN_SPLIT_GRID[row["fn_split_program"]]
        )
        assert (
            row["fn_split_grid_t"]
            == loader.fn_split_grid(arch, 1)
            == loader.FN_SPLIT_GRID[row["fn_split_program_t"]]
        )
        # the column-slab column: the host rule at (T, SM count, arch) -- its token-count bands ahead of the
        # per-architecture program --, the per-architecture program on the no-launch row
        assert row["bwd_slab_program"] == (
            loader.bwd_slab_program(T, sm, arch) if T else loader.BWD_SLAB_PROGRAM[arch]
        )
        if T == 0:
            assert row["bwd_reduce_program"] == "" and row["bwd_reduce_grid"] == 0
        else:
            assert row["bwd_reduce_program"] == loader.bwd_reduce_program(T, sm, arch)
            assert (
                row["bwd_reduce_grid"]
                == loader.bwd_reduce_grid(T, sm, arch)
                == loader.BWD_REDUCE_GRID[row["bwd_reduce_program"]]
            )
            assert row["bwd_reduce_program"] in REDUCE_LIBRARY
        assert (
            row["fn_split_program"] in SPLIT_LIBRARY
            and row["fn_split_program_t"] in SPLIT_LIBRARY
        )
        assert row["bwd_slab_program"] in SLAB_LIBRARY


@pytest.mark.cpu_only
def test_dispatch_of_record_at_the_boundaries() -> None:
    """The kernel-tests reference master's dispatch at every band boundary (:data:`DISPATCH_EXPECTED`: P_K and K units,
    the mapping-pass, coefficient-pass, token-owner, column-slab and reduction programs with the reduction's grid, the
    token-owner rows per CTA) on the four boards, and its per-architecture tables (:data:`TABLES`), reproduced by the registry."""
    assert len(DISPATCH_EXPECTED) == len(DISPATCH_T) * len(BOARDS)
    assert tuple(sorted(set(DISPATCH_T))) == DISPATCH_T
    for (
        T,
        arch,
        sm,
        p_k,
        units,
        fwd_map,
        fwd_coef,
        R,
        bwd_token,
        bwd_slab,
        bwd_reduce,
        grid,
    ) in DISPATCH_EXPECTED:
        assert (arch, sm) in BOARDS and T in DISPATCH_T
        assert loader.k1_units(T, sm, arch)[:2] == (p_k, units), (T, arch, sm)
        assert loader.fwd_map_program(T, sm, arch) == fwd_map, (T, arch, sm)
        assert loader.fwd_coef_program(T, sm, arch) == fwd_coef, (T, arch, sm)
        assert loader.k6_grid(T, sm)[1] == R, (T, sm)
        assert loader.bwd_token_program(T, sm, arch) == bwd_token, (T, arch, sm)
        assert loader.bwd_slab_program(T, sm, arch) == bwd_slab, (T, arch, sm)
        assert loader.bwd_reduce_program(T, sm, arch) == bwd_reduce, (T, arch, sm)
        assert (
            loader.bwd_reduce_grid(T, sm, arch)
            == grid
            == loader.BWD_REDUCE_GRID[bwd_reduce]
        ), (T, arch, sm)
        assert fwd_map in MAP_LIBRARY and fwd_coef in COEF_LIBRARY, (T, arch, sm)
        assert bwd_token in TOKEN_LIBRARY and bwd_reduce in REDUCE_LIBRARY, (
            T,
            arch,
            sm,
        )
        assert bwd_slab in SLAB_LIBRARY, (T, arch, sm)
        # the P_K band of sm_103a (8192 < T <= 16384 -> 16) is the one departure from the cost model's P_K
        if arch == "sm_103a" and 8192 < T <= 16384:
            assert p_k == 16, T
        else:
            assert p_k in (8, 16, 32) and p_k == loader.k1_units(T, sm, "sm_100a")[0], (
                T,
                arch,
                sm,
            )
    for name, value in TABLES.items():
        assert getattr(loader, name) == value, name
    assert loader.K5_GRID_BANDS == {
        "sm_100a": ((0, 1024, 2),),
        "sm_103a": ((0, 4096, 2),),
        "sm_107a": (),
    }
    assert (
        TABLES["K5_FORM_BANDS"] == loader.K5_FORM_BANDS
        and TABLES["K5_FORM_ROWS"] == loader.K5_FORM_ROWS
    )
    assert (
        set(loader.K5_FORM_ROWS) == {"production", *K5_FORM_LIBRARY}
        and loader.K5_FORM_ROWS["production"] == 2
    )

    def _k5(
        T, sm, arch
    ):  # the rule of record spelled from the tables: the form's one-trip grid, the production cap
        form = next(
            (
                f
                for a, b, f in TABLES["K5_FORM_BANDS"][arch]
                if a < T and (b is None or b >= T)
            ),
            "production",
        )
        grid = -(-T // TABLES["K5_FORM_ROWS"][form])
        if form == "production":
            for a, b, g in TABLES["K5_GRID_BANDS"][arch]:
                if a < T and (b is None or b >= T):
                    grid = min(grid, g * sm)
        return form, grid

    for arch, sm in BOARDS:
        for T in (1, 591, 607, 1024, 1025, 4096, 4097, 8192, 8193, 16263, 65536):
            form, grid = _k5(T, sm, arch)
            assert loader.k5_program(T, sm, arch) == form, (arch, sm, T)
            assert loader.k5_grid(T, sm, arch) == grid, (arch, sm, T)
    # spelled out: on 148 SMs the production form's 2 x 148 persistent CTAs to 1024 tokens, then the one-token form's
    # one-trip grid T to 8192 tokens; the production form's ceil(T / 2) above every band; the one-token form from the
    # first token on sm_107a
    assert [loader.k5_grid(T, 148, "sm_100a") for T in (1, 591, 1024, 1025, 4096)] == [
        1,
        296,
        296,
        1025,
        4096,
    ]
    assert (
        loader.k5_grid(65536, 152, "sm_103a") == 32768
        and loader.k5_grid(1024, 208, "sm_107a") == 1024
    )
    assert tuple(loader.ARCHES) == ARCHES
    assert loader.PROGRAM_RULES == (
        "fwd_coef_program",
        "bwd_token_program",
        "bwd_slab_program",
        "bwd_reduce_program",
    )
    for table in (
        "PK_BANDS", "FN_SPLIT_PROGRAM", "FWD_MAP_BANDS", "FWD_MAP_FIT", "FWD_COEF_REGIMES", "BWD_TOKEN_PROGRAM",
        "BWD_TOKEN_REGIMES", "BWD_SLAB_PROGRAM", "BWD_SLAB_BANDS", "BWD_REDUCE_PROGRAM", "BWD_REDUCE_REGIMES",
        "K5_GRID_BANDS", "K5_FORM_BANDS", "BWD_TOKEN_BANDS", "BWD_REDUCE_PARTS", "FWD_FINISH_PROGRAM", "PDL_OFF_BANDS",
    ):  # fmt: skip
        assert tuple(getattr(loader, table)) == ARCHES, table
    assert set(loader.FN_SPLIT_GRID) == {
        p for pair in loader.FN_SPLIT_PROGRAM.values() for p in pair
    }  # one grid per routed program
    for arch in ARCHES:
        assert (
            len(loader.FN_SPLIT_PROGRAM[arch]) == 2
        )  # (the fnp route's program, the fnTp route's program)
        for want_t in (0, 1):
            assert (
                loader.fn_split_program(arch, want_t)
                == loader.FN_SPLIT_PROGRAM[arch][want_t]
            )
            assert (
                SPLIT_LIBRARY[loader.fn_split_program(arch, want_t)] in LIBRARY_ARCHES
            )
            # the route's launch grid: 256 CTAs of 256 threads, or 512 CTAs of 128 for the 128-thread form (both layouts alike)
            assert (
                loader.fn_split_grid(arch, want_t)
                == loader.FN_SPLIT_GRID[loader.fn_split_program(arch, want_t)]
                in (256, 512)
            )
        assert SLAB_LIBRARY[loader.BWD_SLAB_PROGRAM[arch]] in LIBRARY_ARCHES
        for _above, _at_most, program in loader.BWD_SLAB_BANDS[arch]:
            assert SLAB_LIBRARY[program] in LIBRARY_ARCHES, (arch, program)
    # the column-slab bands of record: the dynamic-schedule form (dynr) to 4096 tokens on sm_103a and from 4097 to
    # 8192 tokens on sm_100a; xrsg elsewhere, and at every token count on sm_107a
    assert loader.BWD_SLAB_BANDS == {
        "sm_100a": ((4096, 8192, "dynr"),),
        "sm_103a": ((0, 4096, "dynr"),),
        "sm_107a": (),
    }
    assert [
        loader.bwd_slab_program(T, 152, "sm_103a") for T in (1, 4096, 4097, 65536)
    ] == ["dynr", "dynr", "xrsg", "xrsg"]
    assert [loader.bwd_slab_program(T, 148, "sm_100a") for T in (1, 4096, 4097, 8192, 8193)] == [
        "xrsg", "xrsg", "dynr", "dynr", "xrsg"
    ]  # fmt: skip
    assert all(
        loader.bwd_slab_program(T, sm, "sm_107a") == "xrsg"
        for T in DISPATCH_T
        for sm in (208, 216)
    )
    with pytest.raises(KeyError):
        loader.k1_units(16263, 148, "sm_120a")
    with pytest.raises(KeyError):
        loader.fwd_map_program(16263, 148, "sm_90a")
    with pytest.raises(KeyError):
        loader.fwd_coef_program(16263, 148, "sm_90a")
    with pytest.raises(KeyError):
        loader.bwd_token_program(16263, 148, "sm_90a")
    with pytest.raises(KeyError):
        loader.bwd_reduce_program(16263, 148, "sm_90a")
    with pytest.raises(KeyError):
        loader.fn_split_program("sm_90a", 0)
    with pytest.raises(IndexError):
        loader.fn_split_program("sm_100a", 2)


def _literals() -> dict:
    """The registry's top-level literal assignments, read the way the exporter reads them."""
    source = Path(loader.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        try:
            found[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return found


@pytest.mark.cpu_only
def test_registry_tables_are_literals() -> None:
    literals = _literals()
    for name in (
        "MODULES",
        "ROUTES",
        "HIDDEN_SIZES",
        "ARCHES",
        "WORKSPACE_REGIONS",
        "RECORD_SCALARS",
        "PK_BANDS",
        "FN_SPLIT_PROGRAM",
        "FN_SPLIT_GRID",
        "FWD_MAP_BANDS",
        "FWD_MAP_FIT",
        "FWD_COEF_REGIMES",
        "BWD_TOKEN_PROGRAM",
        "BWD_TOKEN_REGIMES",
        "BWD_SLAB_PROGRAM",
        "BWD_SLAB_BANDS",
        "BWD_REDUCE_PROGRAM",
        "BWD_REDUCE_REGIMES",
        "BWD_REDUCE_GRID",
        "BWD_REDUCE_PARTS",
        "BWD_TOKEN_BANDS",
        "FWD_FINISH_PROGRAM",
        "FWD_FINISH_ROWS",
        "K5_GRID_BANDS",
        "K5_FORM_BANDS",
        "K5_FORM_ROWS",
        "PDL_PATHS",
        "PDL_EDGES",
        "PDL_OFF_BANDS",
    ):
        assert name in literals, f"{name} is not a literal the exporter can read"
    assert literals["MODULES"] == loader.MODULES
    assert literals["ROUTES"] == loader.ROUTES
    assert literals["RECORD_SCALARS"] == loader.RECORD_SCALARS
    assert tuple(literals["HIDDEN_SIZES"]) == (4096,)
    assert tuple(literals["ARCHES"]) == ARCHES


@pytest.mark.cpu_only
def test_records_are_consistent() -> None:
    modules = loader.MODULES
    assert len(modules) == len(RECORD_ARCHES) == N_RECORDS
    libraries = sorted({record["library"] for record in modules.values()})
    assert len(libraries) == N_LIBRARIES and all(
        name.startswith("cake_mhc_train_") for name in libraries
    )
    assert libraries == LIBRARIES
    assert set(libraries) == {
        *SPLIT_LIBRARY.values(), *MAP_LIBRARY.values(), *COEF_LIBRARY.values(), *TOKEN_LIBRARY.values(),
        *SLAB_LIBRARY.values(), *REDUCE_LIBRARY.values(), *FINISH_LIBRARY.values(), *EVERYWHERE,
        *K5_FORM_LIBRARY.values(), *RUN_TIME_ITERS,
    }  # fmt: skip
    for library, names in MULTI_RECORDS.items():
        # several records, or one record not named after its library (the `__hi` / `__t` records)
        assert names != [library]
        assert [
            name for name, record in modules.items() if record["library"] == library
        ] == names, library
    for library in set(libraries) - set(MULTI_RECORDS):
        assert [
            name for name, record in modules.items() if record["library"] == library
        ] == [library], library
    routed = {
        arch: {name for names in loader.ROUTES[arch].values() for name in names}
        for arch in loader.ARCHES
    }
    for name, record in modules.items():
        assert record["kernel"] == record["library"], name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        # A record ships for exactly the architectures whose routes name it (the MODULES trim); a library
        # ships for the union over its records (the exporter's fold).
        record_arches = tuple(arch for arch in loader.ARCHES if name in routed[arch])
        assert tuple(record["arches"]) == record_arches == RECORD_ARCHES[name], name
        library_arches = tuple(
            arch
            for arch in loader.ARCHES
            if any(
                other in routed[arch]
                for other, orec in modules.items()
                if orec["library"] == record["library"]
            )
        )
        assert library_arches == LIBRARY_ARCHES[record["library"]], name
        assert set(record["arches"]) <= set(library_arches), name
        if _is(name, "cake_mhc_train_bwd_prepare"):
            form = K5_FORM_OF_LIBRARY[
                record["library"]
            ]  # the CTA form the record launches (its library)
            assert record["grid_rule"]["kind"] == "rows_per_cta", name
            assert (
                record["grid_rule"]["rows_per_cta"] == TABLES["K5_FORM_ROWS"][form]
            ), name
            # the production record of a live K5_GRID_BANDS band (a bounded rows_between with the band's bounds) carries
            # the persistent cap, the others none (a band the form table covers has no record)
            rule_ = record["select_rule"]
            live = (
                form == "production"
                and rule_ is not None
                and rule_["at_most"] is not None
                and any(
                    (a, b) == (rule_["above"], rule_["at_most"]) and g > 0
                    for arch in record["arches"]
                    for a, b, g in TABLES["K5_GRID_BANDS"][arch]
                )
            )
            assert (record["grid_rule"]["grid_per_sm"] > 0) == live, name
        assert record["hidden"] == 4096 and record["ffi_entry"] == "run", name
        assert (
            record["launch"]["cluster"] == (1, 1, 1)
            and record["launch"]["cooperative"] is False
        ), name
        assert (
            record["launch"]["block"][0] % 32 == 0
            and record["launch"]["dynamic_smem_bytes"] >= 0
        ), name
        rule = record["select_rule"]
        if rule is not None:
            assert rule["kind"] in loader.SELECT_RULE_KINDS, name
            if rule["kind"] == "chunk_rows_band":
                assert set(rule) == {"kind", "ranges"} and _is(
                    name, "cake_mhc_train_bwd_token"
                ), name
                assert all(
                    len(span) == 2 and span[0] >= 0 for span in rule["ranges"]
                ), name
            elif rule["kind"] == "run_time_iters":
                # the run-time-count records: named after their library, launched at every count but the compiled one
                assert (
                    set(rule) == {"kind"}
                    and name in RUN_TIME_ITERS
                    and name == record["library"]
                ), name
                assert _is(name, RUN_TIME_ITERS[name]), name
                assert record["logical"].get("sinkhorn_iters") == "sinkhorn_iters", name
            elif rule["kind"] == "program":
                assert (
                    set(rule) == {"kind", "rule", "program"}
                    and rule["rule"] in loader.PROGRAM_RULES
                ), name
                kernel = next(p for p in PROGRAM_RULE_KERNELS if _is(name, p))
                table, rule_name = PROGRAM_RULE_KERNELS[kernel]
                assert (
                    rule["rule"] == rule_name
                    and table[rule["program"]] == record["library"]
                ), name
                # the K1 and K7 records are the plain ones; a K6 / K8 program record takes the architecture suffix
                # (`__a3` sm_100a, `__a7` sm_107a) where the plain name is another architecture's record under another
                # rule (sm_100a's token-owner forms beside sm_103a's R-range records; sm_107a's beside both)
                assert name in (
                    record["library"],
                    record["library"] + "__a3",
                    record["library"] + "__a7",
                ), name
            else:
                assert set(rule) >= {"kind", "above", "at_most"}, name
            if rule["kind"] == "units_fit":
                assert set(rule) == {"kind", "above", "at_most", "fit"} and _is(
                    name, "cake_mhc_train_fwd_map"
                ), name
            if rule["kind"] == "ctas_band":
                assert set(rule) == {
                    "kind", "above", "at_most", "tile_rows", "lo_ctas_per_sm", "lo_inclusive", "hi_ctas_per_sm",
                    "hi_inclusive",
                }, name  # fmt: skip
                assert rule["tile_rows"] == 8 and _is(
                    name, "cake_mhc_train_fwd_coef"
                ), name
                assert rule["lo_ctas_per_sm"] in (None, 1, 10, 12) and rule[
                    "hi_ctas_per_sm"
                ] in (None, 1, 10, 12)
            if rule["kind"] == "rows_between":
                assert set(rule) == {"kind", "above", "at_most"}, name
                assert _is(
                    name,
                    "cake_mhc_train_fwd_map",
                    "cake_mhc_train_bwd_reduce",
                    "cake_mhc_train_bwd_prepare",
                ), name
        # the kernel parameters carry logical names; the grid triple and the per-call ``enable_pdl`` switch of a
        # ``use_pdl="runtime"`` record are the call's own arguments (no logical name)
        parameters = [
            parameter
            for role, parameter in record["arg_plan"]
            if role not in ("grid", "launch")
        ]
        assert [
            parameter for role, parameter in record["arg_plan"] if role == "grid"
        ] == ["grid_x", "grid_y", "grid_z"]
        launch_slots = [
            parameter for role, parameter in record["arg_plan"] if role == "launch"
        ]
        assert launch_slots == (
            ["enable_pdl"] if record["launch"]["use_pdl"] == "runtime" else []
        ), name
        assert set(parameters) == set(record["logical"]), name
        for parameter, logical in record["logical"].items():
            assert (
                logical in loader.LOGICAL_TENSORS or logical in loader.LOGICAL_SCALARS
            ), (name, parameter, logical)
        roles = {parameter: role for role, parameter in record["arg_plan"]}
        for parameter in parameters:
            logical = record["logical"][parameter]
            expected = "buffer" if logical in loader.LOGICAL_TENSORS else "parameter"
            assert roles[parameter] == expected, (name, parameter)
    # The records of one library agree on what the exporter folds per library (the launch facts); a library ships
    # for the union of its records' architectures (a record ships where its routes name it).
    for library in libraries:
        group = [record for record in modules.values() if record["library"] == library]
        assert set().union(*(set(record["arches"]) for record in group)) == set(
            LIBRARY_ARCHES[library]
        ), library
        assert len({repr(record["launch"]) for record in group}) == 1, library
    for arch, routes in loader.ROUTES.items():
        assert arch in loader.ARCHES
        assert set(routes) == {loader.route_key(4096, kind) for kind in loader.KINDS}
        for kind in loader.KINDS:
            names = routes[loader.route_key(4096, kind)]
            assert names and all(name in modules for name in names), (arch, kind)
            assert all(arch in modules[name]["arches"] for name in names), (arch, kind)
            assert tuple(names) == loader.route_modules(
                arch, 4096, kind
            )  # the literal is a list; the helper a tuple
    # Every route names the records of the libraries the rules of record dispatch on its architecture, in kernel
    # order: the backward route its fn split, its token-owner records, its column-slab records and its reduction
    # records; the forward routes the fn split, the mapping-pass records, the coefficient-pass records (and the
    # collapse on fwd_prepare).  A library's records on one architecture are the ones its routes name.
    for arch in loader.ARCHES:
        split = SPLIT_LIBRARY[
            loader.fn_split_program(arch, 1)
        ]  # the fnTp route (bwd_finish)
        split_fwd = SPLIT_LIBRARY[
            loader.fn_split_program(arch, 0)
        ]  # the fnp route (fwd_prepare / fwd_recompute)
        libs = lambda names: [modules[n]["library"] for n in names]  # noqa: E731
        sms = [sm for a, sm in BOARDS if a == arch]
        if loader.BWD_TOKEN_BANDS[
            arch
        ]:  # the program-rule token-owner records of the band architecture
            tokens = [
                TOKEN_LIBRARY[p]
                for p in _first_programs(loader.bwd_token_program, arch, sms)
            ]
        else:
            tokens = [
                TOKEN_LIBRARY[p] for p, _, _ in loader.BWD_TOKEN_REGIMES[arch]
            ] + [TOKEN_LIBRARY[loader.BWD_TOKEN_PROGRAM[arch]]]
        if loader.BWD_SLAB_BANDS[
            arch
        ]:  # the program-rule column-slab records of a band architecture, first-T order
            slabs = [
                SLAB_LIBRARY[p]
                for p in _first_programs(loader.bwd_slab_program, arch, sms)
            ]
        else:
            slabs = [SLAB_LIBRARY[loader.BWD_SLAB_PROGRAM[arch]]]
        reduces = _first_programs(loader.bwd_reduce_program, arch, sms)
        # the token-owner run-time-count record closes the token-owner records
        assert libs(loader.ROUTES[arch]["bwd_finish_h4096"]) == [
            split, *tokens, "cake_mhc_train_bwd_token__rt64", *slabs, *[REDUCE_LIBRARY[p] for p in reduces]
        ], arch  # fmt: skip
        assert loader.ROUTES[arch]["bwd_finish_h4096"][0] == f"{split}__t"
        # the fn split records launch their program's fixed grid (the rules' fn_split_grid / fn_split_grid_t columns)
        assert modules[f"{split}__t"]["grid_rule"] == {
            "kind": "fixed",
            "ctas": loader.fn_split_grid(arch, 1),
        }, arch
        assert modules[split_fwd]["grid_rule"] == {
            "kind": "fixed",
            "ctas": loader.fn_split_grid(arch, 0),
        }, arch
        maps = [
            n
            for n in libs(loader.ROUTES[arch]["fwd_prepare_h4096"])
            if n.startswith("cake_mhc_train_fwd_map")
        ]
        coefs = [
            n
            for n in libs(loader.ROUTES[arch]["fwd_prepare_h4096"])
            if n.startswith("cake_mhc_train_fwd_coef")
        ]
        assert set(maps) == {
            MAP_LIBRARY[loader.fwd_map_program(T, sm, arch)]
            for T in DISPATCH_T
            for sm in SM_COUNTS
        }, arch
        # the coefficient-pass records the rules dispatch, then the run-time-count record
        assert coefs[-1] == "cake_mhc_train_fwd_coef__rt64", arch
        assert set(coefs[:-1]) == {
            COEF_LIBRARY[loader.fwd_coef_program(T, sm, arch)]
            for T in DISPATCH_T
            for sm in SM_COUNTS
        }, arch
        assert (
            loader.ROUTES[arch]["fwd_prepare_h4096"][0]
            == loader.ROUTES[arch]["fwd_recompute_h4096"][0]
            == split_fwd
        )
        assert (
            loader.ROUTES[arch]["fwd_prepare_h4096"][-1]
            == "cake_mhc_train_fwd_collapse"
        )
        assert loader.ROUTES[arch]["fwd_finish_h4096"] == [FINISH_OF[arch]], arch
        assert (
            loader.ROUTES[arch]["fwd_prepare_h4096"][:-1]
            == loader.ROUTES[arch]["fwd_recompute_h4096"]
        )
        for name in routed[arch]:
            assert arch in LIBRARY_ARCHES[modules[name]["library"]], (arch, name)
        for library, arches in LIBRARY_ARCHES.items():
            assert (arch in arches) == any(
                modules[n]["library"] == library for n in routed[arch]
            ), (arch, library)
    # The fn split runs under one record per (form, layout) some architecture routes: the forward operand layout
    # (want_t 0) and the transposed one of the backward (want_t 1); the production form
    # carries the forward layout alone (every architecture) and the 16-byte-store forms the transposed layout alone
    # (tv128 on sm_100a / sm_103a, its 128-thread form tv128_t128 on sm_107a).
    assert loader.RECORD_SCALARS == RECORD_SCALARS
    for name, scalars in loader.RECORD_SCALARS.items():
        assert modules[name]["role"] == (
            "fn_split_fwd" if scalars["want_t"] == 0 else "fn_split_bwd"
        ), name
        assert _is(name, "cake_mhc_train_fn_split")
    assert {n for n in modules if _is(n, "cake_mhc_train_fn_split")} == set(
        RECORD_SCALARS
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
def test_every_call_selects_one_record_per_dispatched_kernel(
    arch: str, sm: int
) -> None:
    """``selected_modules`` keeps exactly one fn-split, one mapping-pass and one coefficient-pass record per forward
    call and exactly one fn-split, one token-owner, one column-slab and one reduction record per backward call, the
    ones the rules of record name, on every token count up to 70000 and at every band boundary."""
    modules = loader.MODULES

    def only(selected, *prefixes):
        found = [n for n in selected if _is(n, *prefixes)]
        assert len(found) == 1, (arch, sm, selected, prefixes)
        return modules[found[0]]["library"]

    for T in (*range(1, 70001, 97), *DISPATCH_T):
        for kind in ("fwd_prepare", "fwd_recompute"):
            selected = loader.selected_modules(arch, 4096, kind, rows=T, sm_count=sm)
            assert (
                only(selected, "cake_mhc_train_fn_split")
                == SPLIT_LIBRARY[loader.fn_split_program(arch, 0)]
            )
            assert (
                only(selected, "cake_mhc_train_fwd_map")
                == MAP_LIBRARY[loader.fwd_map_program(T, sm, arch)]
            ), (arch, sm, T)
            assert (
                only(selected, "cake_mhc_train_fwd_coef")
                == COEF_LIBRARY[loader.fwd_coef_program(T, sm, arch)]
            ), (arch, sm, T)
            others = [
                n
                for n in selected
                if not _is(
                    n,
                    "cake_mhc_train_fn_split",
                    "cake_mhc_train_fwd_map",
                    "cake_mhc_train_fwd_coef",
                )
            ]
            assert others == (
                ["cake_mhc_train_fwd_collapse"] if kind == "fwd_prepare" else []
            ), (arch, sm, T, kind)
            assert list(selected) == [
                n for n in loader.route_modules(arch, 4096, kind) if n in selected
            ]
        selected = loader.selected_modules(
            arch, 4096, "bwd_finish", rows=T, sm_count=sm
        )
        assert (
            only(selected, "cake_mhc_train_fn_split")
            == SPLIT_LIBRARY[loader.fn_split_program(arch, 1)]
        )
        assert (
            only(selected, "cake_mhc_train_bwd_token")
            == TOKEN_LIBRARY[loader.bwd_token_program(T, sm, arch)]
        ), (arch, sm, T)
        assert (
            only(selected, "cake_mhc_train_bwd_slab")
            == SLAB_LIBRARY[loader.bwd_slab_program(T, sm, arch)]
        ), (arch, sm, T)
        assert (
            only(selected, "cake_mhc_train_bwd_reduce")
            == REDUCE_LIBRARY[loader.bwd_reduce_program(T, sm, arch)]
        ), (arch, sm, T)
        assert len(selected) == 4 and list(selected) == [
            n for n in loader.route_modules(arch, 4096, "bwd_finish") if n in selected
        ]
        assert loader.selected_modules(
            arch, 4096, "fwd_finish", rows=T, sm_count=sm
        ) == loader.route_modules(arch, 4096, "fwd_finish")
        # the backward prepare: exactly one record -- of the library of the CTA form k5_program names (the 128-thread
        # one-token form inside a K5_FORM_BANDS band, the 256-thread record elsewhere); its grid rule reproduces k5_grid
        selected = loader.selected_modules(
            arch, 4096, "bwd_prepare", rows=T, sm_count=sm
        )
        form = loader.k5_program(T, sm, arch)
        assert (
            len(selected) == 1
            and only(selected, "cake_mhc_train_bwd_prepare") == K5_FORM_LIBRARY[form]
        ), (arch, sm, T)
        rule5 = modules[selected[0]]["grid_rule"]
        want5 = -(-T // rule5["rows_per_cta"])
        want5 = min(want5, rule5["grid_per_sm"] * sm) if rule5["grid_per_sm"] else want5
        assert want5 == loader.k5_grid(T, sm, arch), (arch, sm, T, selected)
        assert list(selected) == [
            n for n in loader.route_modules(arch, 4096, "bwd_prepare") if n in selected
        ]
    with pytest.raises(ValueError, match="unknown Cake mHC training selection rule"):
        loader._selects({"kind": "other", "above": 0, "at_most": None}, 1, sm, arch)


@pytest.mark.cpu_only
def test_pdl_only_where_the_kernel_waits_for_the_preceding_grid() -> None:
    """``use_pdl`` is safe only for kernels that wait (``griddepcontrol.wait``) before reading a predecessor's
    output: the coefficient pass, the row streams, the token-owner pass, the column-slab pass (every role waits
    before its first read) and the final reduction do; the fn split (every record: one per form and layout some
    architecture routes) and the mapping pass (every form) do not and must be launched without it.  The dependents of
    a kernel that releases its dependents right after its own wait (the libraries of ``PDL_EDGES``: the collapse and the
    backward prepare after the coefficient pass, the column-slab pass after the token-owner pass, the final reduction
    after the column-slab pass) take the attribute per call (``"runtime"``)."""
    pdl = {name: record["launch"]["use_pdl"] for name, record in loader.MODULES.items()}
    runtime = {
        name
        for name, record in loader.MODULES.items()
        if loader.pdl_edge(record["library"]) is not None
    }
    assert pdl == {
        name: (
            "runtime"
            if name in runtime
            else not _is(name, "cake_mhc_train_fn_split", "cake_mhc_train_fwd_map")
        )
        for name in loader.MODULES
    }
    assert runtime == {name for name in loader.MODULES if _is(name, *loader.PDL_EDGES)}
    n_map = sum(1 for name in loader.MODULES if _is(name, "cake_mhc_train_fwd_map"))
    assert (
        sum(1 for value in pdl.values() if value is False)
        == len(RECORD_SCALARS) + n_map
    )
    # One fn split record per (form, layout) some architecture routes: every fn split record
    # names its layout (``want_t``) and a library of SPLIT_LIBRARY, and every architecture's routes reference exactly
    # one record of each layout -- the forward one (want_t 0) opens fwd_prepare / fwd_recompute, the transposed one
    # (want_t 1) opens bwd_finish.
    assert {
        name for name in loader.MODULES if _is(name, "cake_mhc_train_fn_split")
    } == set(RECORD_SCALARS)
    assert all(scalars.get("want_t") in (0, 1) for scalars in RECORD_SCALARS.values())
    assert {loader.MODULES[name]["library"] for name in RECORD_SCALARS} == set(
        SPLIT_LIBRARY.values()
    )
    for arch in loader.ARCHES:
        splits = {
            route: [
                name for name in loader.ROUTES[arch][route] if name in RECORD_SCALARS
            ]
            for route in (
                "fwd_prepare_h4096",
                "fwd_recompute_h4096",
                "bwd_finish_h4096",
            )
        }
        assert all(len(names) == 1 for names in splits.values()), (arch, splits)
        assert splits["fwd_prepare_h4096"] == splits["fwd_recompute_h4096"], arch
        assert RECORD_SCALARS[splits["fwd_prepare_h4096"][0]] == {"want_t": 0}, arch
        assert RECORD_SCALARS[splits["bwd_finish_h4096"][0]] == {"want_t": 1}, arch
    assert n_map >= len(MAP_LIBRARY)


@pytest.mark.cpu_only
def test_pdl_bands_of_record() -> None:
    """The per-call PDL attribute is off exactly inside the measured token-count bands of ``PDL_OFF_BANDS`` -- per
    architecture, edge (predecessor -> dependent) and call path -- and on in every other call; the records on an edge
    take the ``enable_pdl`` argument right before the grid, no other record takes one."""
    assert loader.PDL_PATHS == ("forward", "backward", "checkpoint")
    assert loader.PDL_OFF_BANDS == PDL_OFF_BANDS
    assert tuple(loader.PDL_OFF_BANDS) == ARCHES
    assert set(loader.PDL_EDGES) == {
        "cake_mhc_train_fwd_collapse", "cake_mhc_train_bwd_prepare", "cake_mhc_train_bwd_slab", "cake_mhc_train_bwd_reduce"
    }  # fmt: skip
    rows_of_record = {row["T"] for row in _rows()}
    for arch, edges in loader.PDL_OFF_BANDS.items():
        for edge, paths in edges.items():
            assert edge in loader.PDL_EDGES.values(), (arch, edge)
            for path, bands in paths.items():
                assert path in loader.PDL_PATHS, (arch, edge, path)
                assert bands and list(bands) == sorted(bands), (arch, edge, path, bands)
                # a tuple of (above, at_most] pairs -- never a bare pair, which the switch would unpack as integers
                assert all(
                    isinstance(band, tuple) and len(band) == 2 for band in bands
                ), (arch, edge, path, bands)
                for index, (above, at_most) in enumerate(bands):
                    assert above < at_most and {above, at_most} <= rows_of_record, (
                        arch,
                        edge,
                        path,
                        above,
                        at_most,
                    )
                    assert index == 0 or bands[index - 1][1] <= above, (
                        arch,
                        edge,
                        path,
                        bands,
                    )
    for name, record in loader.MODULES.items():
        plan = [tuple(entry) for entry in record["arg_plan"]]
        slots = [index for index, (role, _) in enumerate(plan) if role == "launch"]
        assert plan[-3:] == [
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ], name
        if record["launch"]["use_pdl"] == "runtime":
            assert slots == [len(plan) - 4] and plan[-4] == ("launch", "enable_pdl"), (
                name
            )
            assert loader.pdl_edge(record["library"]) is not None, name
        else:
            assert not slots, name
    for arch in ARCHES:
        for name, record in loader.MODULES.items():
            if arch not in record["arches"]:
                continue
            edge = loader.pdl_edge(record["library"])
            for path in loader.PDL_PATHS:
                bands = loader.PDL_OFF_BANDS[arch].get(edge or "", {}).get(path, ())
                boundaries = {
                    t for a, b in bands for t in (a, a + 1, b, b + 1)
                }  # on / off / off / on around a band
                for T in sorted((rows_of_record | boundaries) - {0}):
                    off = record["launch"]["use_pdl"] == "runtime" and any(
                        a < T <= b for a, b in bands
                    )
                    assert loader.enable_pdl(name, T, arch, path) == (
                        0 if off else 1
                    ), (name, arch, path, T)
    assert loader.enable_pdl("cake_mhc_train_fn_split", 1024, "sm_103a", "forward") == 1
    with pytest.raises(ValueError, match="call path"):
        loader.enable_pdl("cake_mhc_train_fwd_collapse", 1024, "sm_103a", "other")


@pytest.mark.cpu_only
def test_launch_facts_of_record() -> None:
    """The block and the dynamic shared memory every module launches with (:data:`LAUNCH_FACTS`: the kernel's
    shared-memory pool plus the 1024-byte control prefix (TMA mbarriers, TMEM hold) where the kernel declares one;
    the exporter's render is the authority, these are the figures of record): the mapping-pass forms at 192 threads
    with a 4 / 3 / 5 x 28672 + 1024 pool (two CTAs per SM, one for the 5-stage form), the lane-packed coefficient
    forms at 128 threads (the production chain at 512), the token-owner pass at 256 threads (90112 + 1024 for every
    form but the TMA ring, whose pool is the render's figure), the column-slab pass at 384 threads (its pool + the
    1024 B reserve within an SM's 228 KiB), the final reduction at its form's block (256 / 64 / 32 threads) and pool
    (the 64 KiB staging tile, the split parameter form's 4224 B, the 64- / 32-thread split forms' 2688 / 2432 B),
    the fn split at 256 threads (128 for the 128-thread 16-byte-store form, whose grid doubles) and no dynamic pool."""
    facts = {
        name: (
            record["launch"]["block"],
            record["launch"]["dynamic_smem_bytes"],
            record["launch"]["persistent_ctas_per_sm"],
        )
        for name, record in loader.MODULES.items()
    }
    assert facts == LAUNCH_FACTS
    for name in loader.RECORD_SCALARS:
        # the fn split covers 65536 thread-slots of 8 elements: 256 CTAs x 256 threads, or 512 x 128 for the 128-thread form
        ctas = loader.MODULES[name]["grid_rule"]["ctas"]
        assert facts[name] == ((65536 // ctas, 1, 1), 0, 1) and ctas in (256, 512), name
    for library, names in MULTI_RECORDS.items():
        assert len({facts[name] for name in names}) == 1, (
            library
        )  # the records of a library agree on their launch
    pools = {4 * 28672 + 1024: 2, 3 * 28672 + 1024: 2, 5 * 28672 + 1024: 1}
    for name in (n for n in loader.MODULES if _is(n, "cake_mhc_train_fwd_map")):
        block, smem, pcs = facts[name]
        assert block == (192, 1, 1) and smem in pools and pcs == pools[smem], name
    for name, record in loader.MODULES.items():
        if _is(name, "cake_mhc_train_fwd_coef"):
            want = (
                ((512, 1, 1), 0, 1)
                if record["library"] == "cake_mhc_train_fwd_coef"
                else None
            )
            if want is None:
                assert facts[name][0] == (128, 1, 1) and facts[name][2] == 1, name
            else:
                assert facts[name] == want, name
    for program, name in TOKEN_LIBRARY.items():
        if program in (
            "ao15_rae8_tma_ntr",
            "ao22_rae7_pd1_tma_ntr",
            "ao22_rae7_pd1_fs_ntr",
            "ao22_rae7_pd1_fs_l2l_ntr",
        ):
            # the TMA rings' and the fold-mbarrier forms' pools are the render's figures (LAUNCH_FACTS pins them)
            assert facts[name][0] == (256, 1, 1) and facts[name][2] == 2, name
            continue
        assert facts[name] == ((256, 1, 1), 91136, 2), name
    for name in SLAB_LIBRARY.values():
        assert facts[name][0] == (384, 1, 1) and facts[name][2] == 1, name
        assert facts[name][1] + 1024 <= 228 * 1024, name
    reduce_facts = {
        "f64": ((256, 1, 1), 65536, 1),
        "s27": ((256, 1, 1), 4224, 1),
        "pt64": ((64, 1, 1), 2688, 1),
        "pt32": ((32, 1, 1), 2432, 1),
    }
    for program, name in REDUCE_LIBRARY.items():
        if program in (
            "wpu4",
            "s27_wcvt_spred",
            "s27_wcvt_fadd_fshfl",
            "wpu4_fadd",
            "wpu4_spred",
        ):
            # the sm_107a and sm_100a per-P-band reduction forms: their pools are the render's figures (LAUNCH_FACTS)
            assert facts[name][0] == (256, 1, 1) and facts[name][2] == 1, name
            continue
        assert facts[name] == reduce_facts[program], name
    # Two mapping-pass CTAs per SM need 2 x (115712 + 1024 B reserve) = the 228 KiB of an SM100-class SM; the
    # 5-stage form runs one CTA per SM.
    assert 2 * (115712 + 1024) <= 228 * 1024
    assert 5 * 28672 + 1024 + 1024 <= 228 * 1024


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
def test_launch_grids_follow_the_rules(
    monkeypatch: pytest.MonkeyPatch, arch: str, sm: int
) -> None:
    monkeypatch.setattr(loader, "sm_count", lambda device_index: sm)
    monkeypatch.setattr(loader, "device_arch", lambda device_index: arch)
    modules = loader.MODULES
    for row in _rows():
        T = row["T"]
        if T == 0 or (row["sm_count"], row["arch"]) != (sm, arch):
            continue
        grid = lambda name: loader.launch_grid(modules[name], T, device_index=0)  # noqa: E731
        for name in loader.RECORD_SCALARS:
            assert grid(name) == (modules[name]["grid_rule"]["ctas"], 1, 1), name
        for want_t, route, column in (
            (0, "fwd_prepare_h4096", "fn_split_grid"),
            (1, "bwd_finish_h4096", "fn_split_grid_t"),
        ):
            selected = loader.ROUTES[arch][route][0]  # the route's fn split record
            assert (
                grid(selected)
                == (row[column], 1, 1)
                == (loader.fn_split_grid(arch, want_t), 1, 1)
            ), (arch, selected)
        for name in (n for n in modules if _is(n, "cake_mhc_train_fwd_map")):
            persistent = modules[name]["grid_rule"].get("ctas_per_sm")
            if persistent:
                # persistent CTAs per SM: min(ctas_per_sm x SMs, units); equal to k1_grid exactly where the record
                # is selected
                assert grid(name) == (min(persistent * sm, row["k1_units"]), 1, 1), name
                if loader._selects(modules[name]["select_rule"], T, sm, arch):
                    assert grid(name) == (row["k1_grid"], 1, 1), name
                continue
            assert grid(name) == (row["k1_grid"], 1, 1), name
        program_of = {library: program for program, library in COEF_LIBRARY.items()}
        for name in (n for n in modules if _is(n, "cake_mhc_train_fwd_coef")):
            program = program_of.get(modules[name]["library"])
            tokens = 8 if name in RUN_TIME_ITERS else COEF_TOKENS_PER_CTA[program]
            assert grid(name) == (-(-T // tokens), 1, 1), name
        assert grid(COEF_LIBRARY[row["fwd_coef_program"]]) == (
            -(-T // COEF_TOKENS_PER_CTA[row["fwd_coef_program"]]),
            1,
            1,
        )
        # One trip per CTA (the production forms): two column halves per token for the collapse, four tokens
        # times two halves per CTA for the residual mixing, two tokens per CTA for the backward prepare.
        assert (
            grid("cake_mhc_train_fwd_collapse")
            == (row["k3_grid"], 1, 1)
            == (2 * T, 1, 1)
        )
        assert (
            grid(FINISH_OF[row["arch"]])
            == (row["k4_grid"], 1, 1)
            == (loader.k4_grid_of_record(T, row["arch"]), 1, 1)
            == (
                -(
                    -(2 * T)
                    // loader.FWD_FINISH_ROWS[loader.fwd_finish_program(row["arch"])]
                ),
                1,
                1,
            )
        )
        # the backward prepare: every record's own rule (the form's one-trip grid ceil(T / rows_per_cta), or
        # min(ceil(T / 2), grid_per_sm x SMs) on the production band record); the record the call selects launches the
        # row's k5_grid and is of the library of the row's CTA form (k5_program)
        for name in (n for n in modules if _is(n, "cake_mhc_train_bwd_prepare")):
            per_sm = modules[name]["grid_rule"]["grid_per_sm"]
            trip = -(-T // modules[name]["grid_rule"]["rows_per_cta"])
            want = min(trip, per_sm * sm) if per_sm else trip
            assert grid(name) == (want, 1, 1), name
        (selected,) = loader.selected_modules(
            arch, 4096, "bwd_prepare", rows=T, sm_count=sm
        )
        assert (
            grid(selected)
            == (row["k5_grid"], 1, 1)
            == (loader.k5_grid(T, sm, arch), 1, 1)
        )
        form = loader.k5_program(T, sm, arch)
        assert (
            row["k5_program"] == form
            and modules[selected]["library"] == K5_FORM_LIBRARY[form]
        ), (arch, T, selected)
        rows5 = TABLES["K5_FORM_ROWS"][form]
        bands = (
            [
                (a, b, g)
                for a, b, g in loader.K5_GRID_BANDS[arch]
                if a < T and (b is None or b >= T)
            ]
            if form == "production"
            else []
        )
        if bands:
            assert len(bands) == 1, (arch, T, bands)
            _, at_most, per_sm = bands[0]
            assert row["k5_grid"] == min(-(-T // rows5), per_sm * sm), (arch, sm, T)
            assert selected == f"cake_mhc_train_bwd_prepare__g{per_sm}sm_le{at_most}", (
                arch,
                T,
                selected,
            )
        else:
            assert row["k5_grid"] == -(-T // rows5), (arch, sm, T)
            assert selected == K5_FORM_LIBRARY[form] or selected.startswith(
                K5_FORM_LIBRARY[form] + "__"
            ), (arch, T, selected)
        for name in (*TOKEN_LIBRARY.values(), "cake_mhc_train_bwd_token__rt64"):
            assert grid(name) == (row["G6"], 1, 1), name
        for name in SLAB_LIBRARY.values():
            assert grid(name) == (row["k7_grid"], 1, 1), name
        for program, name in REDUCE_LIBRARY.items():
            assert grid(name) == (REDUCE_GRID[program], 1, 1), name
        assert grid(REDUCE_LIBRARY[row["bwd_reduce_program"]]) == (
            row["bwd_reduce_grid"],
            1,
            1,
        )
        for name in (n for n in modules if _is(n, "cake_mhc_train_bwd_slab")):
            assert grid(name) == (row["k7_grid"], 1, 1), name
    some_coef = next(n for n in modules if _is(n, "cake_mhc_train_fwd_coef"))
    with pytest.raises(ValueError):
        loader.launch_grid(modules[some_coef], 0, device_index=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
def test_workspace_regions_and_envelope(arch: str, sm: int) -> None:
    names = {
        "fwd": ["ticket", "fnp", "spart", "qpart"],
        "bwd": ["ticket", "fnTp", "dzp", "tokp", "param_partial", "dfn_part"],
    }
    column = {"fwd": "fwd_workspace_bytes", "bwd": "bwd_workspace_bytes"}
    for row in _rows():
        if (row["sm_count"], row["arch"]) != (sm, arch):
            continue
        T = row["T"]
        for kind in ("fwd", "bwd"):
            layout = loader.workspace_regions(kind, T, sm, arch)
            assert (
                layout["total_bytes"]
                == row[column[kind]]
                == loader.workspace_bytes(kind, T, sm, arch)
            )
            if T == 0:
                assert layout["regions"] == {} and layout["total_bytes"] == 0
                continue
            assert list(layout["regions"]) == names[kind]
            offset = 0
            for region in layout["regions"].values():
                assert region["offset"] == offset and offset % 256 == 0
                assert region["bytes"] > 0
                offset += (
                    -(-region["bytes"] // 256) * 256
                )  # each region is padded to the next 256-byte boundary
            assert offset == layout["total_bytes"]
            assert (
                layout["regions"]["ticket"]["shape"] == (64,)
                and layout["regions"]["ticket"]["dtype"] == "uint32"
                and layout["regions"]["ticket"]["offset"] == 0
            )  # the same ticket words for every token count a buffer serves
            assert (
                loader.workspace_bytes_upto(kind, T, sm, arch) >= layout["total_bytes"]
            )
    for kind in ("fwd", "bwd"):
        previous = -1
        for T in range(0, 4200):
            envelope = loader.workspace_bytes_upto(kind, T, sm, arch)
            assert envelope >= previous and envelope >= loader.workspace_bytes(
                kind, T, sm, arch
            ), (kind, T)
            previous = envelope
    with pytest.raises(ValueError):
        loader.workspace_regions("fwd", -1, sm, arch)
    with pytest.raises(ValueError):
        loader.workspace_regions("other", 1, sm, arch)


@pytest.mark.cpu_only
def test_the_p_k_band_changes_the_forward_workspace_on_sm_103a_alone() -> None:
    """The P_K = 16 band (8192 < T <= 16384 on sm_103a) doubles the K-unit partials of the forward workspace; the
    backward workspace and the other architectures are unchanged."""
    for T in (8192, 8193, 16263, 16384, 16385):
        same = loader.workspace_bytes("fwd", T, 148, "sm_100a")
        assert loader.workspace_bytes("fwd", T, 148, "sm_107a") == same
        band = loader.workspace_bytes("fwd", T, 148, "sm_103a")
        if 8192 < T <= 16384:
            assert band > same, T
        else:
            assert band == same, T
        for arch in loader.ARCHES:
            assert loader.workspace_bytes(
                "bwd", T, 148, arch
            ) == loader.workspace_bytes("bwd", T, 148, "sm_100a")


@pytest.mark.cpu_only
def test_route_glue() -> None:
    assert loader.supported_hidden_sizes("sm_100a") == (4096,)
    assert loader.arch_for_capability((10, 0)) == "sm_100a"
    assert loader.arch_for_capability((10, 3)) == "sm_103a"
    assert loader.arch_for_capability((10, 7)) == "sm_107a"
    assert loader.arch_for_capability((9, 0)) is None
    assert loader.route_applies(
        device_capability=(10, 3), hidden=4096, kind="bwd_finish"
    )
    assert not loader.route_applies(
        device_capability=(12, 0), hidden=4096, kind="fwd_prepare"
    )
    assert not loader.route_applies(
        device_capability=(10, 0), hidden=2048, kind="fwd_prepare"
    )
    assert (
        loader.workspace_kind("fwd_prepare")
        == loader.workspace_kind("fwd_recompute")
        == "fwd"
    )
    assert (
        loader.workspace_kind("bwd_finish") == "bwd"
        and loader.workspace_kind("fwd_finish") is None
    )
    for arch in loader.ARCHES:
        assert loader.selected_modules(
            arch, 4096, "fwd_finish", rows=129, sm_count=148
        ) == loader.route_modules(arch, 4096, "fwd_finish")
        for kind in ("fwd_prepare", "fwd_recompute", "bwd_finish", "bwd_prepare"):
            selected = loader.selected_modules(arch, 4096, kind, rows=129, sm_count=148)
            route = loader.route_modules(arch, 4096, kind)
            # a route with band records selects a strict subset; a route of unconditional records selects every record
            banded = any(
                loader.MODULES[name]["select_rule"] is not None for name in route
            )
            assert set(selected) <= set(route)
            assert (set(selected) < set(route)) == banded, (arch, kind)
        selected = loader.selected_modules(
            arch, 4096, "fwd_prepare", rows=129, sm_count=148
        )
        libraries = {loader.MODULES[name]["library"] for name in selected}
        want = loader.fwd_coef_program(129, 148, arch)
        assert (
            COEF_LIBRARY[want] in libraries
            and len(libraries & set(COEF_LIBRARY.values())) == 1
        )
        assert MAP_LIBRARY[loader.fwd_map_program(129, 148, arch)] in libraries
    # library_name: the library's architecture suffix where the record ships for the architecture, a refusal elsewhere.
    for name, record in loader.MODULES.items():
        for arch in loader.ARCHES:
            if arch in record["arches"]:
                assert (
                    loader.library_name(name, arch) == f"{record['library']}_{arch}"
                ), (name, arch)
            else:
                with pytest.raises(RuntimeError, match=f"not delivered for {arch}"):
                    loader.library_name(name, arch)


@pytest.mark.cpu_only
def test_run_time_count_rule_and_bound() -> None:
    assert loader.SINKHORN_ITERS_MAX == 64
    assert "run_time_iters" in loader.SELECT_RULE_KINDS


@pytest.mark.cpu_only
def test_selection_by_sinkhorn_count() -> None:
    """At 20 no run-time-count record is selected. At another count the coefficient
    and token-owner passes select exactly their run-time-count record, and every
    other kernel the default's record, on every board and route."""
    run_time = {
        name
        for name, record in loader.MODULES.items()
        if (record["select_rule"] or {}).get("kind") == "run_time_iters"
    }
    assert run_time == set(RUN_TIME_ITERS)
    roles = {loader.MODULES[name]["role"]: name for name in run_time}
    assert set(roles) == {"fwd_coef", "bwd_token"}
    for arch, sm in BOARDS:
        for kind in loader.KINDS:
            for T in (1, 7, 129, 1024, 2049, 4096, 8192, 14928, 16263, 32768, 65536):
                default = loader.selected_modules(arch, 4096, kind, rows=T, sm_count=sm)
                assert not set(default) & run_time, (arch, kind, T)
                assert default == loader.selected_modules(
                    arch, 4096, kind, rows=T, sm_count=sm, sinkhorn_iters=20
                )
                for iters in (1, 19, 21, 64):
                    other = loader.selected_modules(
                        arch, 4096, kind, rows=T, sm_count=sm, sinkhorn_iters=iters
                    )
                    expected = tuple(
                        roles.get(loader.MODULES[name]["role"], name)
                        for name in default
                    )
                    assert other == expected, (arch, kind, T, iters)
                    assert other == tuple(
                        n for n in loader.route_modules(arch, 4096, kind) if n in other
                    )
