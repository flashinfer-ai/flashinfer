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

Host plan and loader glue of the Cake four-stream mHC training kernels with
the non-deterministic backward (hidden sizes 4096, 5120 and 7168, 4 streams;
SM100 / SM103, and SM107 at hidden 4096).

This registry is a self-contained copy of the deterministic package's registry
(``flashinfer.cake_mhc_det_train_backend.cake_jit``): every record, route, host
rule and table below is that registry's under the ``cake_mhc_train_`` prefix,
except what the backward's ``bwd_finish`` route changes.  The deterministic
column-slab records (``bwd_slab``) and final-reduction records (``bwd_reduce``)
are absent with their selection tables and their PDL edge (``K7->K8``), and
four programs of this package take their place:

* ``cake_mhc_train_fn_split_atomic__tv128_dep`` (sm_100a, sm_103a) and
  ``cake_mhc_train_fn_split_atomic__tv128_t128_dep`` (sm_107a): the
  deterministic transposed-layout fn split of the architecture (16-byte
  ``fnTp`` stores on 256- / 128-thread CTAs), bit for bit, which with
  ``accumulate == 0`` also zeroes the FP32 ``dfn`` output, launched after the
  token-owner pass as its programmatic dependent (``_dep``): it releases its
  own dependent first and waits for the token-owner pass as its last
  instruction, so it runs beside that pass (it reads only ``fn`` and writes
  only ``fnTp`` and ``dfn``, which the token-owner pass never touches) and
  completes only after it;
* ``cake_mhc_train_bwd_slab_atomic__dynr`` (every architecture): the
  deterministic dynamic-schedule column-slab form ``dynr`` (``dX`` bit for
  bit), which adds the ``dfn`` sum of every (column slab, token chunk) unit into
  ``dfn`` with FP32 reductions (``red.global.add.f32``; units without tokens add
  nothing) instead of writing one partial slot per token chunk, and folds the
  parameter gradients ``dscale`` / ``dbase`` from the token-owner pass's per-CTA
  partials in one warp of each of the first 27 CTAs (the deterministic
  reduction's sequence: an FP64 chain in ascending order, the carried-in value
  last, one rounding), so they are bitwise the deterministic package's;
* ``cake_mhc_train_bwd_slab_atomic__lpt`` (every architecture): the same
  column slab under the two-size tail schedule (``schedule="lpt"``).  It runs
  ``p_big`` chunks of ``rpc`` tokens first, then ``P - p_big`` smaller chunks
  of ``rpc_small`` tokens.  Each next unit is claimed when the CTA loads the
  last tile of its current unit, so the small units end the pass on whichever
  SMs are free.  Every unit is the dynamic form's unit of the same tokens
  (``dX`` bit for bit, the same parameter fold).
  :func:`bwd_slab_atomic_program` picks it per token count and device where
  :func:`lpt_chunks` (calibrated per architecture by ``BWD_SLAB_LPT`` at hidden
  4096, per architecture and size by ``BWD_SLAB_LPT_HIDDEN`` at 5120 and 7168)
  returns a geometry, and ``dynr`` elsewhere.

At hidden 5120 and 7168 (sm_100a, sm_103a) the atomic fn split and the dynamic
column slab have one more record per size under the deterministic suffix
``__h<H>`` (``cake_mhc_train_fn_split_atomic__tv128_dep__h<H>``,
``cake_mhc_train_bwd_slab_atomic__dynr__h<H>``), and on sm_100a the two-size
column slab too (``cake_mhc_train_bwd_slab_atomic__lpt__h<H>``), picked as at
hidden 4096.  sm_103a has no ``lpt`` calibration of these sizes yet
(``BWD_SLAB_LPT_HIDDEN`` is ``None`` there): its column slab runs ``dynr`` at
every token count.

So ``bwd_finish`` is the deterministic token-owner pass, the fn split and the
column slab: no final reduction, and the ``bwd`` workspace has no per-chunk
``dfn`` partial region (``dfn_part``).  Only the summation order of ``dfn``
varies from run to run: each element sums at most ``P`` per-chunk partials
(``P <= 32``: the chunk count of :func:`k7_chunks`, or of the two-size geometry
under ``lpt``, whose chunks can cover other token ranges, so its total can also
differ from the deterministic ``dfn`` in the low-order bits), so it lies within
``P`` FP32 roundings of the fixed-order sum of those partials, and the FP32
reduction flushes subnormal addends and results to zero.  ``dX``, ``dscale``,
``dbase`` and every other route's outputs are the deterministic package's bits.
The deterministic package's backward fn splits (``fn_split__tv128__t``,
``fn_split__tv128_t128__t``, ``fn_split__tv128__h<H>__t``) have no copy here:
the atomic fn splits replace them.

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch,
grid, selection and workspace rules and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is the ordered list of module records one public call launches:
``fwd_prepare`` (fn split, mapping GEMM, coefficients, collapse), ``fwd_recompute``
(the strict prefix of ``fwd_prepare`` that regenerates the per-token statistics
bitwise for checkpoint replay), ``fwd_finish`` (residual mixing), ``bwd_prepare``
(``dB`` and the per-token gradient statistics) and ``bwd_finish`` (transposed fn
split, token-owner pass, column-slab pass).  The token count ``T`` is a runtime
scalar: a changed number of rows never recompiles anything, and ``T == 0``
launches nothing.

A module record is one (kernel, launch policy) pair.  A kernel launched under
several policies has one record per policy (``<kernel>__<stage>`` after the
first), all naming the kernel's ``library``, so it is compiled once.

The grid, chunk and workspace rules are pure integer functions of ``(T, SM
count, architecture, hidden size)`` (:func:`k1_units` with the measured P_K band of
``PK_BANDS``, :func:`k7_chunks`, :func:`k6_grid`, :func:`k2_grid`,
:func:`k3_grid`, :func:`k4_grid`, :func:`k5_grid`; :func:`row_grid` is the
persistent row grid of the design table, kept for the rules table), the
deterministic package's rules verbatim, so the work partition is a pure
function of ``(T, device)`` and the same as the deterministic package's.  The
column slab's two-size geometry is this package's own rule
(:func:`lpt_chunks`, with its launch values :func:`slab_plan`), still a
pure function of ``(T, SM count, architecture, hidden size)``; where
it fires, the column slab's chunks differ from the deterministic
chunks.  Which program a call launches is a rule of the same kind
(rounds 2-7): :func:`fwd_map_program`, :func:`fwd_coef_program` and
:func:`bwd_token_program` name the mapping-pass, coefficient-pass and
token-owner forms of record per token count and board (the mapping pass's
schedule form by its K units per CTA: ``FWD_MAP_CLAIM``), spelled on the records
as ``select_rule`` (``rows_between`` / ``units_fit`` / ``ctas_band`` /
``chunk_rows_band`` / ``program``); :func:`fn_split_program` names the fn
split's form per architecture and route (``FN_SPLIT_PROGRAM[arch] = (fnp,
fnTp)``; the forward operand layout ``want_t = 0`` and the transposed layout of
the backward, whose form the atomic fn split of the architecture keeps) with
its launch grid :func:`fn_split_grid` (``FN_SPLIT_GRID`` per program:
H / 16 CTAs, 256 at hidden 4096, and H / 8 for the 128-thread form);
the column-slab pass's form is :func:`bwd_slab_atomic_program`:
``dynr``, or ``lpt`` where :func:`lpt_chunks` returns a geometry (every
architecture at hidden 4096, sm_100a at 5120 and 7168); the residual
mixing's program of record is per architecture
too (``FWD_FINISH_PROGRAM`` / :func:`fwd_finish_program`, its
launch grid :func:`k4_grid_of_record`), and the token-owner band
``BWD_TOKEN_BANDS`` (sm_107a) is read by the rule itself. The backward prepare's
CTA form is :func:`k5_program` ``(T, SM count, architecture)``
(``K5_FORM_BANDS``: the 128-thread one-token form ``__t128`` on sm_100a ``1024 < T <= 8192``, sm_103a ``T <= 8192``, sm_107a ``T <= 65536``,
the 256-thread two-token record elsewhere) and its launch grid :func:`k5_grid`:
the form's one-trip grid, or the record's persistent 2-CTAs-per-SM grid inside a
band of ``K5_GRID_BANDS`` (sm_100a ``T <= 1024``) -- the band
record's ``grid_per_sm``.

Every record names its hidden size (``hidden``, one of :data:`HIDDEN_SIZES`).
The hidden sizes 5120 and 7168 are the deterministic package's (libraries
``__h<H>`` on the routes ``<kind>_h5120`` / ``<kind>_h7168``, on sm_100a and
sm_103a; sm_107a serves 4096 only), copied under the same rule as hidden 4096:
the deterministic ``__h<H>`` records but the backward fn split, the column slab
and the reduction, and a ``bwd_finish`` of the deterministic token-owner
records, the atomic fn split ``__h<H>`` and the column slab ``dynr__h<H>``
(and ``lpt__h<H>`` on sm_100a).
Every rule takes the hidden size as its last argument and reads its tables
through ``_tab`` as the deterministic registry's do (``H_TABLES`` spells every
table for 5120 and 7168: a size it does not spell is an error, never the 4096
value; the column slab's screen reads ``BWD_SLAB_LPT_HIDDEN`` instead), and
the workspace regions sized by the flattened token (``fnp``, ``fnTp``) are
spelled in ``n = 4 H``.

Programmatic dependent launch is a launch attribute of the generated binding,
recorded per module as ``launch.use_pdl``: ``True`` for the kernels whose device
code waits for the preceding grid (``griddepcontrol.wait``) before reading a
predecessor's output -- the coefficient pass, the row streams, the token-owner
pass and the column-slab pass -- and for the backward fn split, which waits for
the token-owner pass as its last instruction; ``False`` for the forward fn
splits and the mapping pass, which carry no such wait.  The dependents of a
kernel that releases its dependents right after its own wait (the collapse and
the backward prepare after the coefficient pass) or at its start (the
column-slab pass after the backward fn split) record ``"runtime"``: their
binding takes ``enable_pdl`` per call, and :func:`enable_pdl` leaves the
attribute off inside the measured token-count bands of ``PDL_OFF_BANDS`` (per
architecture, edge and call path; the deterministic package's bands of those
edges), where the early launch would hold the dependent's CTAs resident while
the predecessor still runs.  The column-slab pass (both forms) keeps the
deterministic ``K6->K7`` bands: the token-owner pass releases the fn split at
once and the fn split releases the column-slab pass at once, so the column-slab
pass's early launch overlaps the token-owner pass as in the deterministic route.
As in the deterministic package, the bands were measured at hidden 4096 and
apply at every hidden size.
"""

from __future__ import annotations

import functools
import heapq
import math
from typing import Any, Optional, Sequence

# Filled mechanically from the complete verified program bundle (the derived
# leaves ``kernel_symbol``, ``arg_plan``, ``logical`` and the launch leaves
# ``block`` / ``cluster`` / ``cooperative`` / ``dynamic_smem_bytes`` /
# ``use_pdl``); everything else is the host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_mhc_train_fn_split": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnp"],
            ["buffer", "fnTp"],
            ["parameter", "want_t"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fn_split",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fn_split",
        "logical": {
            "fn": "fn",
            "fnTp": "fnTp",
            "fnp": "fnp",
            "want_t": "want_t",
        },
        "role": "fn_split_fwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fn_split_atomic__tv128_dep": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnTp"],
            ["buffer", "dfn"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fn_split_atomic__tv128_dep",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split_atomic__tv128_dep",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fn_split_atomic__tv128_dep",
        "logical": {
            "accumulate": "accumulate",
            "dfn": "dfn",
            "fn": "fn",
            "fnTp": "fnTp",
        },
        "role": "fn_split_bwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fn_split_atomic__tv128_t128_dep": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnTp"],
            ["buffer", "dfn"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 512,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fn_split_atomic__tv128_t128_dep",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split_atomic__tv128_t128_dep",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fn_split_atomic__tv128_t128_dep",
        "logical": {
            "accumulate": "accumulate",
            "dfn": "dfn",
            "fn": "fn",
            "fnTp": "fnTp",
        },
        "role": "fn_split_bwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_st5_mb1": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 1,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_st5_mb1",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 144384,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "above": 0,
            "at_most": 8191,
            "fit": True,
            "kind": "units_fit",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx__tk": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "tk_nsx_ct",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_ct": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_ct",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_ct",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 114816,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_ct",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_ct",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_g2_ol2l": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_g2_ol2l",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 114944,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "tk_nsx_g2_ol2l_ct",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_st5_mb1__a3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 1,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_st5_mb1",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 144384,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "above": 0,
            "at_most": 8192,
            "fit": True,
            "kind": "units_fit",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__ns96_g2_ol2l": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__ns96_g2_ol2l",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__ns96_g2_ol2l",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__ns96_g2_ol2l",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "ns96_g2_ol2l",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__ns96_g2_ol2l__tk": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__ns96_g2_ol2l",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__ns96_g2_ol2l",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__ns96_g2_ol2l",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "tk_ns96_g2_ol2l_ct",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__ns96_g2_ol2l_ct": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__ns96_g2_ol2l_ct",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__ns96_g2_ol2l_ct",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 114816,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__ns96_g2_ol2l_ct",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "ns96_g2_ol2l_ct",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__a3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_g2_ol2l",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_st5_mb1__a7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 1,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_st5_mb1",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 144384,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_st5_mb1",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "above": 0,
            "at_most": 32768,
            "fit": True,
            "kind": "units_fit",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__a7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_g2_ol2l",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "above": 0,
            "at_most": 32768,
            "fit": False,
            "kind": "units_fit",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__hi": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10_tree": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10_tree",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10_tree",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10_tree",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10_tree",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st16_w4_tree": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 4,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__st16_w4_tree",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st16_w4_tree",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st16_w4_tree",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st16_w4_tree",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12800,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st16_lp2_as_el_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10_pu4": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10_pu4",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10_pu4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10_pu4",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10_pu4",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__lp2_w8_adj": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__lp2_w8_adj",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__lp2_w8_adj",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__lp2_w8_adj",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "lp2_w8_adj",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2_as_el_qd",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2_as_el_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__pfl2_qmt": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__pfl2_qmt",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__pfl2_qmt",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__pfl2_qmt",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "pfl2_qmt",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__g2_el": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 16,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__g2_el",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__g2_el",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 25600,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__g2_el",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "g2_el",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2_tree_asel_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__lp2_w8_adj_qe": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__lp2_w8_adj_qe",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__lp2_w8_adj_qe",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__lp2_w8_adj_qe",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "lp2_w8_adj_qe",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__rt64": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["parameter", "sinkhorn_iters"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef__rt64",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__rt64",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__rt64",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "sinkhorn_iters": "sinkhorn_iters",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "run_time_iters",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_collapse": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "u"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 2,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_collapse",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_collapse",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_fwd_collapse",
        "logical": {
            "T": "rows",
            "stats": "stats",
            "u": "u",
            "x": "x",
        },
        "role": "fwd_collapse",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_fwd_finish": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "b"],
            ["buffer", "stats"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 2,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_finish",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_finish",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_finish",
        "logical": {
            "T": "rows",
            "b": "b",
            "stats": "stats",
            "x": "x",
            "y": "y",
        },
        "role": "fwd_finish",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_fwd_finish__r2": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "b"],
            ["buffer", "stats"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 2,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_finish__r2",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_finish__r2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_finish__r2",
        "logical": {
            "T": "rows",
            "b": "b",
            "stats": "stats",
            "x": "x",
            "y": "y",
        },
        "role": "fwd_finish",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__g2sm_le1024": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__t128": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare__t128",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare__t128",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 640,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare__t128",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": {
            "above": 0,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__t128__a3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare__t128",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare__t128",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 640,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare__t128",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__t128__a7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare__t128",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare__t128",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 640,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare__t128",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": {
            "above": 0,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__gt8192": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": {
            "above": 8192,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__gt65536": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "chunk_rows_band",
            "ranges": [
                [0, 32],
            ],
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "chunk_rows_band",
            "ranges": [
                [49, 56],
            ],
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__dot_ao8": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__dot_ao8",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__dot_ao8",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__dot_ao8",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "chunk_rows_band",
            "ranges": [
                [33, 48],
                [57, None],
            ],
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao22_rae7_pd1_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao22_rae7_pd1_tma_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__dot_ao8__a3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__dot_ao8",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__dot_ao8",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__dot_ao8",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "dot_ao8",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao15_rae8_tma_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao22_rae7_rf3_pd1_l2l_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao22_rae7_pd1_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__dot_ao8_l2l": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__dot_ao8_l2l",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__dot_ao8_l2l",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__dot_ao8_l2l",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "dot_ao8_l2l",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao15_rae8_tma_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__rt64": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["parameter", "sinkhorn_iters"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token__rt64",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__rt64",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 71296,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__rt64",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "sinkhorn_iters": "sinkhorn_iters",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "run_time_iters",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab_atomic__dynr": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn"],
            ["buffer", "param_partial"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["buffer", "ticket"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k7_chunks",
            "kind": "slab_chunks",
            "slabs": 64,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_slab_atomic__dynr",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab_atomic__dynr",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228480,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_slab_atomic__dynr",
        "logical": {
            "P": "parts",
            "T": "rows",
            "X": "x_flat",
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dbase": "dbase",
            "dfn": "dfn",
            "dscale": "dscale",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "param_partial": "param_partial",
            "rpc": "rows_per_chunk",
            "ticket": "ticket",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": {
            "kind": "program",
            "program": "dynr",
            "rule": "bwd_slab_atomic_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab_atomic__lpt": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn"],
            ["buffer", "param_partial"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["buffer", "ticket"],
            ["parameter", "p_big"],
            ["parameter", "rpc_small"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "lpt_chunks",
            "kind": "slab_chunks",
            "slabs": 64,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_slab_atomic__lpt",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab_atomic__lpt",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228480,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_slab_atomic__lpt",
        "logical": {
            "P": "lpt_parts",
            "T": "rows",
            "X": "x_flat",
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dbase": "dbase",
            "dfn": "dfn",
            "dscale": "dscale",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "p_big": "lpt_big_parts",
            "param_partial": "param_partial",
            "rpc": "lpt_rows_per_chunk",
            "rpc_small": "lpt_small_rows_per_chunk",
            "ticket": "ticket",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": {
            "kind": "program",
            "program": "lpt",
            "rule": "bwd_slab_atomic_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fn_split__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnp"],
            ["buffer", "fnTp"],
            ["parameter", "want_t"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 320,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fn_split__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fn_split__h5120",
        "logical": {
            "fn": "fn",
            "fnTp": "fnTp",
            "fnp": "fnp",
            "want_t": "want_t",
        },
        "role": "fn_split_fwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fn_split_atomic__tv128_dep__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnTp"],
            ["buffer", "dfn"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 320,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fn_split_atomic__tv128_dep__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split_atomic__tv128_dep__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fn_split_atomic__tv128_dep__h5120",
        "logical": {
            "accumulate": "accumulate",
            "dfn": "dfn",
            "fn": "fn",
            "fnTp": "fnTp",
        },
        "role": "fn_split_bwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 1,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 144384,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_st5_mb1",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_g2_ol2l",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10_tree",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st16_w4_tree__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 4,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st16_w4_tree",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12800,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st16_lp2_as_el_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10_pu4",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "lp2_w8_adj",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h5120": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2_as_el_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_collapse__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "u"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 5,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_collapse__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_collapse__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_fwd_collapse__h5120",
        "logical": {
            "T": "rows",
            "stats": "stats",
            "u": "u",
            "x": "x",
        },
        "role": "fwd_collapse",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_fwd_finish__w2__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "b"],
            ["buffer", "stats"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 5,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_finish__w2__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_finish__w2__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_finish__w2__h5120",
        "logical": {
            "T": "rows",
            "b": "b",
            "stats": "stats",
            "x": "x",
            "y": "y",
        },
        "role": "fwd_finish",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__t160__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_bwd_prepare__t160__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare__t160__h5120",
        "launch": {
            "block": (160, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 896,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare__t160__h5120",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 111616,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h5120",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao22_rae7_pd1_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__dot_ao8__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_bwd_token__dot_ao8__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__dot_ao8__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 111616,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__dot_ao8__h5120",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "dot_ao8",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab_atomic__dynr__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn"],
            ["buffer", "param_partial"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["buffer", "ticket"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k7_chunks",
            "kind": "slab_chunks",
            "slabs": 80,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_bwd_slab_atomic__dynr__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab_atomic__dynr__h5120",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228480,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_slab_atomic__dynr__h5120",
        "logical": {
            "P": "parts",
            "T": "rows",
            "X": "x_flat",
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dbase": "dbase",
            "dfn": "dfn",
            "dscale": "dscale",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "param_partial": "param_partial",
            "rpc": "rows_per_chunk",
            "ticket": "ticket",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": {
            "kind": "program",
            "program": "dynr",
            "rule": "bwd_slab_atomic_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab_atomic__lpt__h5120": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn"],
            ["buffer", "param_partial"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["buffer", "ticket"],
            ["parameter", "p_big"],
            ["parameter", "rpc_small"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "lpt_chunks",
            "kind": "slab_chunks",
            "slabs": 80,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_bwd_slab_atomic__lpt__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab_atomic__lpt__h5120",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228480,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_slab_atomic__lpt__h5120",
        "logical": {
            "P": "lpt_parts",
            "T": "rows",
            "X": "x_flat",
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dbase": "dbase",
            "dfn": "dfn",
            "dscale": "dscale",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "p_big": "lpt_big_parts",
            "param_partial": "param_partial",
            "rpc": "lpt_rows_per_chunk",
            "rpc_small": "lpt_small_rows_per_chunk",
            "ticket": "ticket",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": {
            "kind": "program",
            "program": "lpt",
            "rule": "bwd_slab_atomic_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fn_split__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnp"],
            ["buffer", "fnTp"],
            ["parameter", "want_t"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 448,
            "kind": "fixed",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fn_split__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fn_split__h7168",
        "logical": {
            "fn": "fn",
            "fnTp": "fnTp",
            "fnp": "fnp",
            "want_t": "want_t",
        },
        "role": "fn_split_fwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fn_split_atomic__tv128_dep__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnTp"],
            ["buffer", "dfn"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 448,
            "kind": "fixed",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fn_split_atomic__tv128_dep__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split_atomic__tv128_dep__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fn_split_atomic__tv128_dep__h7168",
        "logical": {
            "accumulate": "accumulate",
            "dfn": "dfn",
            "fn": "fn",
            "fnTp": "fnTp",
        },
        "role": "fn_split_bwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 1,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 144384,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_st5_mb1",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": {
            "kind": "program",
            "program": "nsx_g2_ol2l",
            "rule": "fwd_map_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10_tree",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st16_w4_tree__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 4,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st16_w4_tree",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12800,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st16_lp2_as_el_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "qe_mb10_pu4",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "lp2_w8_adj",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h7168": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "program",
            "program": "st8_lp2_as_el_qd",
            "rule": "fwd_coef_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_collapse__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "u"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 7,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_collapse__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_collapse__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_fwd_collapse__h7168",
        "logical": {
            "T": "rows",
            "stats": "stats",
            "u": "u",
            "x": "x",
        },
        "role": "fwd_collapse",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_fwd_finish__w2__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "b"],
            ["buffer", "stats"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 7,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_finish__w2__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_finish__w2__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_finish__w2__h7168",
        "logical": {
            "T": "rows",
            "b": "b",
            "stats": "stats",
            "x": "x",
            "y": "y",
        },
        "role": "fwd_finish",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare__t224__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_bwd_prepare__t224__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare__t224__h7168",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1152,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_prepare__t224__h7168",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 1,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 152576,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "ao22_rae7_pd1_ntr",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_token__dot_ao8__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 1,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_bwd_token__dot_ao8__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__dot_ao8__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 152576,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__dot_ao8__h7168",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "program",
            "program": "dot_ao8",
            "rule": "bwd_token_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab_atomic__dynr__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn"],
            ["buffer", "param_partial"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["buffer", "ticket"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k7_chunks",
            "kind": "slab_chunks",
            "slabs": 112,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_bwd_slab_atomic__dynr__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab_atomic__dynr__h7168",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228480,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_slab_atomic__dynr__h7168",
        "logical": {
            "P": "parts",
            "T": "rows",
            "X": "x_flat",
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dbase": "dbase",
            "dfn": "dfn",
            "dscale": "dscale",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "param_partial": "param_partial",
            "rpc": "rows_per_chunk",
            "ticket": "ticket",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": {
            "kind": "program",
            "program": "dynr",
            "rule": "bwd_slab_atomic_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab_atomic__lpt__h7168": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn"],
            ["buffer", "param_partial"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["buffer", "ticket"],
            ["parameter", "p_big"],
            ["parameter", "rpc_small"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "lpt_chunks",
            "kind": "slab_chunks",
            "slabs": 112,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_bwd_slab_atomic__lpt__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab_atomic__lpt__h7168",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 228480,
            "persistent_ctas_per_sm": 1,
            "use_pdl": "runtime",
        },
        "library": "cake_mhc_train_bwd_slab_atomic__lpt__h7168",
        "logical": {
            "P": "lpt_parts",
            "T": "rows",
            "X": "x_flat",
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dbase": "dbase",
            "dfn": "dfn",
            "dscale": "dscale",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "p_big": "lpt_big_parts",
            "param_partial": "param_partial",
            "rpc": "lpt_rows_per_chunk",
            "rpc_small": "lpt_small_rows_per_chunk",
            "ticket": "ticket",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": {
            "kind": "program",
            "program": "lpt",
            "rule": "bwd_slab_atomic_program",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fwd_coef__rt64__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["parameter", "sinkhorn_iters"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_fwd_coef__rt64__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__rt64__h5120",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__rt64__h5120",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "sinkhorn_iters": "sinkhorn_iters",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "run_time_iters",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_bwd_token__rt64__h5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["parameter", "sinkhorn_iters"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 5120,
        "kernel": "cake_mhc_train_bwd_token__rt64__h5120",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__rt64__h5120",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 71296,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__rt64__h5120",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "sinkhorn_iters": "sinkhorn_iters",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "run_time_iters",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fwd_coef__rt64__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["parameter", "sinkhorn_iters"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 8,
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_fwd_coef__rt64__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef__rt64__h7168",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef__rt64__h7168",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "sinkhorn_iters": "sinkhorn_iters",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": {
            "kind": "run_time_iters",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_bwd_token__rt64__h7168": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["parameter", "sinkhorn_iters"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 1,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 7168,
        "kernel": "cake_mhc_train_bwd_token__rt64__h7168",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token__rt64__h7168",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 71296,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token__rt64__h7168",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "sinkhorn_iters": "sinkhorn_iters",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": {
            "kind": "run_time_iters",
        },
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
}

# Route key -> ordered module records, per architecture.  Every route lists
# every record it may launch; ``selected_modules`` keeps exactly the records
# whose ``select_rule`` admits the call (one fn-split, one mapping-pass and one
# coefficient-pass record per forward call, one fn-split, one token-owner and
# one column-slab record per backward call: the per-T dispatch of rounds 2-7,
# keyed by the architecture through the routes and by the token count and the
# device's SM count through the rules).  The routes are the deterministic
# package's, except ``bwd_finish``: the deterministic token-owner records, the
# atomic fn split of the architecture (launched after the token-owner pass, as
# its programmatic dependent) and the atomic column slab's two forms (``dynr``,
# ``lpt``: one per call by ``bwd_slab_atomic_program``; no reduction record) at
# hidden 4096 and on sm_100a at hidden 5120 and 7168 (``__h<H>``), its ``dynr``
# form alone (``__h<H>``) on sm_103a at hidden 5120 and 7168.
# The launch order is the route order.  A record ships for the architectures of
# its ``arches`` (the sm_103a-only forms for sm_103a alone, the ``lpt__h<H>``
# column slabs for sm_100a alone); the device code is folded over those
# architectures.
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd_finish_h4096": [
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a3",
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr",
            "cake_mhc_train_bwd_token__dot_ao8__a3",
            "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a3",
            "cake_mhc_train_bwd_token__rt64",
            "cake_mhc_train_fn_split_atomic__tv128_dep",
            "cake_mhc_train_bwd_slab_atomic__dynr",
            "cake_mhc_train_bwd_slab_atomic__lpt",
        ],
        "bwd_prepare_h4096": [
            "cake_mhc_train_bwd_prepare__g2sm_le1024",
            "cake_mhc_train_bwd_prepare__t128__a3",
            "cake_mhc_train_bwd_prepare__gt8192",
        ],
        "fwd_finish_h4096": [
            "cake_mhc_train_fwd_finish",
        ],
        "fwd_prepare_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__a3",
            "cake_mhc_train_fwd_map__ns96_g2_ol2l_ct",
            "cake_mhc_train_fwd_map__ns96_g2_ol2l",
            "cake_mhc_train_fwd_map__ns96_g2_ol2l__tk",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__a3",
            "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
            "cake_mhc_train_fwd_coef__st16_w4_tree",
            "cake_mhc_train_fwd_coef__qe_mb10_tree",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4",
            "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__st8_lp2",
            "cake_mhc_train_fwd_coef__lp2_w8_adj",
            "cake_mhc_train_fwd_coef__rt64",
            "cake_mhc_train_fwd_collapse",
        ],
        "fwd_recompute_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__a3",
            "cake_mhc_train_fwd_map__ns96_g2_ol2l_ct",
            "cake_mhc_train_fwd_map__ns96_g2_ol2l",
            "cake_mhc_train_fwd_map__ns96_g2_ol2l__tk",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__a3",
            "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
            "cake_mhc_train_fwd_coef__st16_w4_tree",
            "cake_mhc_train_fwd_coef__qe_mb10_tree",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4",
            "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__st8_lp2",
            "cake_mhc_train_fwd_coef__lp2_w8_adj",
            "cake_mhc_train_fwd_coef__rt64",
        ],
        "bwd_finish_h5120": [
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h5120",
            "cake_mhc_train_bwd_token__dot_ao8__h5120",
            "cake_mhc_train_bwd_token__rt64__h5120",
            "cake_mhc_train_fn_split_atomic__tv128_dep__h5120",
            "cake_mhc_train_bwd_slab_atomic__dynr__h5120",
            "cake_mhc_train_bwd_slab_atomic__lpt__h5120",
        ],
        "bwd_prepare_h5120": [
            "cake_mhc_train_bwd_prepare__t160__h5120",
        ],
        "fwd_finish_h5120": [
            "cake_mhc_train_fwd_finish__w2__h5120",
        ],
        "fwd_prepare_h5120": [
            "cake_mhc_train_fn_split__h5120",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10__h5120",
            "cake_mhc_train_fwd_coef__st8_lp2__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
            "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h5120",
            "cake_mhc_train_fwd_coef__rt64__h5120",
            "cake_mhc_train_fwd_collapse__h5120",
        ],
        "fwd_recompute_h5120": [
            "cake_mhc_train_fn_split__h5120",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10__h5120",
            "cake_mhc_train_fwd_coef__st8_lp2__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
            "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h5120",
            "cake_mhc_train_fwd_coef__rt64__h5120",
        ],
        "bwd_finish_h7168": [
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168",
            "cake_mhc_train_bwd_token__dot_ao8__h7168",
            "cake_mhc_train_bwd_token__rt64__h7168",
            "cake_mhc_train_fn_split_atomic__tv128_dep__h7168",
            "cake_mhc_train_bwd_slab_atomic__dynr__h7168",
            "cake_mhc_train_bwd_slab_atomic__lpt__h7168",
        ],
        "bwd_prepare_h7168": [
            "cake_mhc_train_bwd_prepare__t224__h7168",
        ],
        "fwd_finish_h7168": [
            "cake_mhc_train_fwd_finish__w2__h7168",
        ],
        "fwd_prepare_h7168": [
            "cake_mhc_train_fn_split__h7168",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10__h7168",
            "cake_mhc_train_fwd_coef__st8_lp2__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
            "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h7168",
            "cake_mhc_train_fwd_coef__rt64__h7168",
            "cake_mhc_train_fwd_collapse__h7168",
        ],
        "fwd_recompute_h7168": [
            "cake_mhc_train_fn_split__h7168",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10__h7168",
            "cake_mhc_train_fwd_coef__st8_lp2__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
            "cake_mhc_train_fwd_coef__st8_lp2_as_el_qd__h7168",
            "cake_mhc_train_fwd_coef__rt64__h7168",
        ],
    },
    "sm_103a": {
        "bwd_finish_h4096": [
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr",
            "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr",
            "cake_mhc_train_bwd_token__dot_ao8",
            "cake_mhc_train_bwd_token__rt64",
            "cake_mhc_train_fn_split_atomic__tv128_dep",
            "cake_mhc_train_bwd_slab_atomic__dynr",
            "cake_mhc_train_bwd_slab_atomic__lpt",
        ],
        "bwd_prepare_h4096": [
            "cake_mhc_train_bwd_prepare__t128",
            "cake_mhc_train_bwd_prepare__gt8192",
        ],
        "fwd_finish_h4096": [
            "cake_mhc_train_fwd_finish",
        ],
        "fwd_prepare_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map__nsx_st5_mb1",
            "cake_mhc_train_fwd_map__nsx_ct",
            "cake_mhc_train_fwd_map__nsx",
            "cake_mhc_train_fwd_map__nsx__tk",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l",
            "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
            "cake_mhc_train_fwd_coef__qe_mb10_tree",
            "cake_mhc_train_fwd_coef__st16_w4_tree",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__qe_mb10",
            "cake_mhc_train_fwd_coef__st8_lp2",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4",
            "cake_mhc_train_fwd_coef__lp2_w8_adj",
            "cake_mhc_train_fwd_coef__rt64",
            "cake_mhc_train_fwd_collapse",
        ],
        "fwd_recompute_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map__nsx_st5_mb1",
            "cake_mhc_train_fwd_map__nsx_ct",
            "cake_mhc_train_fwd_map__nsx",
            "cake_mhc_train_fwd_map__nsx__tk",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l",
            "cake_mhc_train_fwd_map__tk_nsx_g2_ol2l_ct",
            "cake_mhc_train_fwd_coef__qe_mb10_tree",
            "cake_mhc_train_fwd_coef__st16_w4_tree",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__qe_mb10",
            "cake_mhc_train_fwd_coef__st8_lp2",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4",
            "cake_mhc_train_fwd_coef__lp2_w8_adj",
            "cake_mhc_train_fwd_coef__rt64",
        ],
        "bwd_finish_h5120": [
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h5120",
            "cake_mhc_train_bwd_token__dot_ao8__h5120",
            "cake_mhc_train_bwd_token__rt64__h5120",
            "cake_mhc_train_fn_split_atomic__tv128_dep__h5120",
            "cake_mhc_train_bwd_slab_atomic__dynr__h5120",
        ],
        "bwd_prepare_h5120": [
            "cake_mhc_train_bwd_prepare__t160__h5120",
        ],
        "fwd_finish_h5120": [
            "cake_mhc_train_fwd_finish__w2__h5120",
        ],
        "fwd_prepare_h5120": [
            "cake_mhc_train_fn_split__h5120",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10__h5120",
            "cake_mhc_train_fwd_coef__st8_lp2__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
            "cake_mhc_train_fwd_coef__rt64__h5120",
            "cake_mhc_train_fwd_collapse__h5120",
        ],
        "fwd_recompute_h5120": [
            "cake_mhc_train_fn_split__h5120",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h5120",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h5120",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10__h5120",
            "cake_mhc_train_fwd_coef__st8_lp2__h5120",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h5120",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h5120",
            "cake_mhc_train_fwd_coef__rt64__h5120",
        ],
        "bwd_finish_h7168": [
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168",
            "cake_mhc_train_bwd_token__dot_ao8__h7168",
            "cake_mhc_train_bwd_token__rt64__h7168",
            "cake_mhc_train_fn_split_atomic__tv128_dep__h7168",
            "cake_mhc_train_bwd_slab_atomic__dynr__h7168",
        ],
        "bwd_prepare_h7168": [
            "cake_mhc_train_bwd_prepare__t224__h7168",
        ],
        "fwd_finish_h7168": [
            "cake_mhc_train_fwd_finish__w2__h7168",
        ],
        "fwd_prepare_h7168": [
            "cake_mhc_train_fn_split__h7168",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10__h7168",
            "cake_mhc_train_fwd_coef__st8_lp2__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
            "cake_mhc_train_fwd_coef__rt64__h7168",
            "cake_mhc_train_fwd_collapse__h7168",
        ],
        "fwd_recompute_h7168": [
            "cake_mhc_train_fn_split__h7168",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__h7168",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_w4_tree__h7168",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10__h7168",
            "cake_mhc_train_fwd_coef__st8_lp2__h7168",
            "cake_mhc_train_fwd_coef__qe_mb10_pu4__h7168",
            "cake_mhc_train_fwd_coef__lp2_w8_adj__h7168",
            "cake_mhc_train_fwd_coef__rt64__h7168",
        ],
    },
    "sm_107a": {
        "bwd_finish_h4096": [
            "cake_mhc_train_bwd_token__ao22_rae7_rf3_pd1_l2l_ntr",
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_tma_ntr",
            "cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__a7",
            "cake_mhc_train_bwd_token__dot_ao8_l2l",
            "cake_mhc_train_bwd_token__ao15_rae8_tma_ntr__a7",
            "cake_mhc_train_bwd_token__rt64",
            "cake_mhc_train_fn_split_atomic__tv128_t128_dep",
            "cake_mhc_train_bwd_slab_atomic__dynr",
            "cake_mhc_train_bwd_slab_atomic__lpt",
        ],
        "bwd_prepare_h4096": [
            "cake_mhc_train_bwd_prepare__t128__a7",
            "cake_mhc_train_bwd_prepare__gt65536",
        ],
        "fwd_finish_h4096": [
            "cake_mhc_train_fwd_finish__r2",
        ],
        "fwd_prepare_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__a7",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__a7",
            "cake_mhc_train_fwd_map__hi",
            "cake_mhc_train_fwd_coef__st16_w4_tree",
            "cake_mhc_train_fwd_coef__pfl2_qmt",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__g2_el",
            "cake_mhc_train_fwd_coef__qe_mb10_tree",
            "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd",
            "cake_mhc_train_fwd_coef__lp2_w8_adj_qe",
            "cake_mhc_train_fwd_coef__rt64",
            "cake_mhc_train_fwd_collapse",
        ],
        "fwd_recompute_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map__nsx_st5_mb1__a7",
            "cake_mhc_train_fwd_map__nsx_g2_ol2l__a7",
            "cake_mhc_train_fwd_map__hi",
            "cake_mhc_train_fwd_coef__st16_w4_tree",
            "cake_mhc_train_fwd_coef__pfl2_qmt",
            "cake_mhc_train_fwd_coef__st16_lp2_as_el_qd",
            "cake_mhc_train_fwd_coef__g2_el",
            "cake_mhc_train_fwd_coef__qe_mb10_tree",
            "cake_mhc_train_fwd_coef__st8_lp2_tree_asel_qd",
            "cake_mhc_train_fwd_coef__lp2_w8_adj_qe",
            "cake_mhc_train_fwd_coef__rt64",
        ],
    },
}
# Every hidden size the registry holds records for.  An architecture serves a
# size when all five routes of that size exist (``supported_hidden_sizes``):
# sm_100a and sm_103a serve all three, sm_107a serves 4096.
HIDDEN_SIZES: tuple[int, ...] = (4096, 5120, 7168)

# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
KINDS = ("fwd_prepare", "fwd_recompute", "fwd_finish", "bwd_prepare", "bwd_finish")
# The routes that hold a caller-owned workspace, by the ``workspace`` name of
# their records' ``workspace_rule``.
WORKSPACE_OF_KIND: dict[str, str] = {
    "fwd_prepare": "fwd",
    "fwd_recompute": "fwd",
    "bwd_finish": "bwd",
}

# Model constants the kernels are built for (``_static_`` dimensions).
STREAMS = 4
# The hidden size of record and its flattened token; every record names its own
# hidden size (``record["hidden"]``, one of ``HIDDEN_SIZES``) and the rules take
# it as their last argument (default ``HIDDEN``).
HIDDEN = 4096
HIDDEN_TOTAL = 16384  # the flattened token: 4 streams x 4096
MAPPING_ROWS = 24  # 4 pre logits + 4 post logits + 16 Sinkhorn logits
N_PAD = 32  # mapping rows padded to the tensor-core N of the mapping GEMMs
PIECES = 3  # BF16 pieces of an FP32 operand (exact three-way split)
SLAB_D = 64  # columns per stream of one column-slab unit of the backward
TILE_T = 128  # token tile of the forward mapping pass (``T_pad`` granularity)
SLAB_TILE_T = 64  # token tile of the column-slab pass
RMS_EPS = 1e-5
HC_EPS = 1e-6
SINKHORN_ITERS = 20
# The Sinkhorn iteration counts a call may ask for: 1 .. SINKHORN_ITERS_MAX.  The count of record
# (SINKHORN_ITERS) runs the compiled programs, whose chain is unrolled; every other count runs the
# run-time-count records (``select_rule`` kind ``run_time_iters``), which take the count as their
# ``sinkhorn_iters`` kernel argument and were built for at most SINKHORN_ITERS_MAX (the token-owner
# pass keeps the 2 x count - 1 normalisation steps of its forward replay in shared memory).
SINKHORN_ITERS_MAX = 64

# Per-token records (FP32 columns).  ``stats`` is retained between the forward
# and the backward (256 bytes per token); ``gstats`` is written by
# ``bwd_prepare`` (128 bytes per token).
STATS_COLS = 64
GSTATS_COLS = 32
TOKP_COLS = 24
DZP_COLS = 96  # three BF16 pieces of dz' = dz * r, 32 columns each
PARAM_COLS = 32
TICKET_WORDS = 64  # one 256-byte region of uint32 per workspace, zero on first use; the opt-in ticket schedule of the mapping pass (forward) and the column-slab forms (backward) draw their units from word 0 and leave it 0 again; the static forms leave it unread
STATS_LAYOUT: dict[str, tuple[int, int]] = {
    "r": (0, 1),
    "z": (1, 25),
    "pre": (25, 29),
    "post": (29, 33),
    "C": (33, 49),
    "pad": (49, 64),
}
GSTATS_LAYOUT: dict[str, tuple[int, int]] = {
    "dpost": (0, 4),
    "dC": (4, 20),
    "pad": (20, 32),
}
TOKP_LAYOUT: dict[str, tuple[int, int]] = {
    "r": (0, 1),
    "c_t": (1, 2),
    "pre": (2, 6),
    "C": (6, 22),
    "pad": (22, 24),
}
PARAM_LAYOUT: dict[str, tuple[int, int]] = {
    "da_sum": (0, 24),
    "dscale_terms": (24, 27),
    "pad": (27, 32),
}

# The caller-owned workspaces (``workspace_rule`` kind ``regions``): ordered
# regions, each rounded up to 256 bytes; a shape entry that is a string names
# a host-plan quantity of ``host_plan`` (``t_pad``, ``p_k``, ``chunks``,
# ``n`` = 4 H, the flattened token of the call's hidden size H).
# The ticket region comes first: its offset is 0 for every token count, so a
# buffer that serves several token counts (a caller buffer sized by
# ``workspace_bytes_upto``, a default workspace of a token-count bucket, a CUDA
# Graph captured at a smaller T) keeps its ticket words where the previous call
# left them -- zero -- instead of reading them from bytes another token count's
# layout used for partials or records.  ``fwd`` is the deterministic package's;
# ``bwd`` is the deterministic package's without its last region, the per-chunk
# FP32 ``dfn`` partials (``dfn_part``, ``parts x 24 x n``): the column slab
# adds into ``dfn`` itself, and every other region keeps its deterministic
# offset, so a deterministic backward workspace of the same hidden size also
# serves this package.
WORKSPACE_REGIONS: dict[str, list[dict[str, Any]]] = {
    "fwd": [
        {"dtype": "uint32", "name": "ticket", "shape": [64]},
        {"dtype": "bfloat16", "name": "fnp", "shape": [3, 32, "n"]},
        {"dtype": "float32", "name": "spart", "shape": ["p_k", "t_pad", 24]},
        {"dtype": "float32", "name": "qpart", "shape": ["p_k", "t_pad"]},
    ],
    "bwd": [
        {"dtype": "uint32", "name": "ticket", "shape": [64]},
        {"dtype": "bfloat16", "name": "fnTp", "shape": [3, "n", 32]},
        {"dtype": "bfloat16", "name": "dzp", "shape": ["t_pad", 96]},
        {"dtype": "float32", "name": "tokp", "shape": ["t_pad", 24]},
        {"dtype": "float32", "name": "param_partial", "shape": ["chunks", 32]},
    ],
}
# Scalar values a record fixes itself (the launch policy of a stage record):
# the forward fn split of each hidden size writes the forward operand layout
# (``want_t = 0``).  The atomic fn splits take no ``want_t``: they write the
# transposed layout of the backward only.
RECORD_SCALARS: dict[str, dict[str, int]] = {
    "cake_mhc_train_fn_split": {
        "want_t": 0,
    },
    "cake_mhc_train_fn_split__h5120": {
        "want_t": 0,
    },
    "cake_mhc_train_fn_split__h7168": {
        "want_t": 0,
    },
}
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).  The
# ``*_flat`` names are the ``[T, 4 H]`` views of the ``[T, 4, H]`` tensors
# (the tensor-core kernels describe them as two-dimensional); ``dzp_rows`` is
# the first ``T`` rows of the padded ``dzp`` region.
LOGICAL_TENSORS = (
    "x",
    "x_flat",
    "fn",
    "fnp",
    "fnTp",
    "scale",
    "base",
    "u",
    "stats",
    "b",
    "y",
    "dy",
    "dy_flat",
    "db",
    "gstats",
    "du",
    "dx_flat",
    "dzp",
    "dzp_rows",
    "tokp",
    "param_partial",
    "spart",
    "qpart",
    "ticket",
    "dfn",
    "dscale",
    "dbase",
)
LOGICAL_SCALARS = (
    "rows",
    "t_pad",
    "m_tiles",
    "log2_pk",
    "p_k",
    "parts",
    "rows_per_chunk",
    "chunks",
    "rows_per_cta",
    "rms_eps",
    "hc_eps",
    "sinkhorn_iters",
    "want_t",
    "accumulate",
    "lpt_parts",
    "lpt_rows_per_chunk",
    "lpt_big_parts",
    "lpt_small_rows_per_chunk",
)
# The host-rule table the kernels' test references and this registry are
# checked against (the same rows, in the same column order).
RULES_SM_COUNTS = (148, 152, 208, 216)
# The (architecture, SM count) boards of record: B200, GB300, the two R200 boards.
RULES_BOARDS = (("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 208), ("sm_107a", 216))
RULES_T_VALUES = (
    0, 1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 255, 1000, 1024, 2049, 4096, 8192,
    14928, 15288, 15566, 15590, 16114, 16132, 16164, 16263, 20001, 32768, 65536,
)  # fmt: skip


# ----------------------------------------------------------------------------
# Host rules (integer arithmetic only; valid for T >= 1).  These are the
# deterministic package's rules verbatim (the text of record shared with the
# kernels' test references): a change here is a change of the work partition.
# The exception is :data:`BWD_SLAB_LPT`, :data:`BWD_SLAB_LPT_HIDDEN`,
# :func:`lpt_chunks`, :func:`bwd_slab_atomic_program` and :func:`slab_plan`,
# this package's own column-slab rule: a list-scheduling screen in floating
# point with the result of the column slab's kernel-test screen (not its text:
# _lpt_makespan takes runs of equal units, the unit-by-unit schedule bit for
# bit), calibrated per architecture by BWD_SLAB_LPT at hidden 4096 and per
# architecture and hidden size by BWD_SLAB_LPT_HIDDEN at 5120 and 7168.


# Rounds 2-7: the per-architecture dispatch of record. sm_103a (GB300) carries the forms paired on GB300 in
# two sessions; sm_100a (B200) carries the forms the cross-architecture pairs kept on B200 (the "last paired-and-kept
# form"); sm_107a (R200, the 3186-MHz HBM class of record) carries the forms its class-of-record sessions kept there.
# Every table below is keyed by the architecture so that the nine copies of this text, the FlashInfer registry and the
# rules table branch in one place; a missing architecture is an error, never a default.
#
# PK_BANDS: (above, at_most, P_K) overrides of the cost model's P_K; on sm_103a P_K = 16 for 8192 < T <= 16384 (the
# model's P_K = 8 leaves 6-7 units per SM, fill 0.88-0.96; 12-13 half units fill 0.95-0.98: K2 0.917-0.959 of
# production, forward kernel-row sum with the kept K1 form 0.906-0.939 of cake_r1 on the eight rows). P_K = 32 at
# T <= 1216 and P_K = 16 at T = 4096 were measured and dropped (the K1 partial fold costs more than K2 gains); the
# P_K-16 reading of the ld_group form at 32768 (one session) is information for a later round.
# On sm_100a the band is REVERT-ON-ARCH (the st3ts @ P_K 16 band rows 14928-15590 slower beyond
# spread in both sessions) and on sm_107a the P_K rule stays cake_r1's (st3ts / pk16 REVERT) -> the cost
# model's P_K everywhere there.
PK_BANDS = {"sm_100a": (), "sm_103a": ((8192, 16384, 16),), "sm_107a": ()}
# FN_SPLIT_PROGRAM: the fn split's program per architecture AND route (T-independent; the pair is indexed by the
# kernel's ``want_t``: [0] = the forward operand layout "fnp" of fwd_prepare / fwd_recompute, [1] = the transposed
# layout "fnTp" of bwd_finish). "b16" = the cake_r1 production store path (one 2-byte store per k of the transposed
# layout), "tv128" = t_store="v128" (16-byte coalesced fnTp stores; bitwise). The two layouts run the same binary, and
# the gain is the fnTp route's alone: on GB300 fnTp 1.077-1.096 of production in two sessions (13 / 13 rows) while the
# fnp rows read <= 1.000 in both (0 / 13 > 1.000); on B200 fnTp 1.072-1.086 (13 / 13 in s1, the four anchor T
# in s2) while the fnp rows read <= 1.000: sm_100a and sm_103a dispatch tv128 on the fnTp route and retain
# cake_r1's production on the fnp route (the fallback per route). sm_107a: the same split
# (a5d, two class-of-record sessions: fnTp tv128 / production 0.9242-0.9431 and 0.9244-0.9390 on 13 / 13 rows,
# every control valid, beyond spread; fnp: no valid row faster in either session, 1.0000-1.0200 / 1.0000-1.0238 ->
# tv128 on both routes is forbidden; the KEEP on the fnp route was one session), refined on the fnTp route to
# "tv128_t128" = t_store="v128" on 128-thread CTAs (threads=128: the tv128 map in twice as many, half-sized CTAs,
# grid 512; bitwise): ON tv128 it reads 0.9507-0.9630 / 0.9512-0.9634 at the 13 T in two R200 class-of-record sessions
# on two nodes, entirely below tv128's paired spread at every T, controls 1.000-1.003 (64-thread CTAs and 4-row lanes
# win too, 32-thread CTAs lose). The fnp route of sm_107a keeps the production program; sm_100a / sm_103a keep tv128
# (the 128-thread form is no entry there).
FN_SPLIT_PROGRAM = {
    "sm_100a": ("b16", "tv128"),
    "sm_103a": ("b16", "tv128"),
    "sm_107a": ("b16", "tv128_t128"),
}
# FN_SPLIT_GRID: the launch grid per fn-split program (both layouts launch the same count; T-independent): 256 CTAs of
# 256 threads for "b16" / "tv128", 512 CTAs of 128 threads for "tv128_t128" -- what mhc_det_fn_split.grid_for_symbol
# returns for the entry's symbol and the kernel-tests launch passes.
FN_SPLIT_GRID = {"b16": 256, "tv128": 256, "tv128_t128": 512}
# FWD_MAP_BANDS: (above, at_most, program) of the mapping pass by token count (at_most None = unbounded); outside every
# band the production form FWD_MAP_PRODUCTION ("ss_st4": a_path="ss", 4 stages). Every form below is EXACT: the
# chunk's second stage issues [b1; b3] as ONE N64 MMA into the adjacent TMEM pair of the layout [big_0 | small | big_1]
# (n_stack="exact"); production's accumulator chains, promotion and Kahan order, hence BITWISE production at the same
# P_K on all 13 T and every gate T. On sm_103a: "nsx" for 1 <= T <= 8191 at the host rule's
# P_K (faster than production on every row in two 13-row sessions on two nodes: 0.9623 / 0.9604 at 1024, 0.9806 /
# 0.9825 at 4096; 4096 keeps nsx: the ld_group / out_l2 forms read 1.0040 / 1.0098 and 0.9945 /
# 1.0003 of nsx there) and "nsx_g2_ol2l" (ld_group=2: the load warp frees and issues two adjacent 64-k stages
# together; out_l2="evict_last" on the spart / qpart stores the coefficient pass reads next) for every T >= 8192: vs
# production 0.9510 / 0.9449 at 32768 and 0.9466 / 0.9395 at 65536 on two nodes, nsx16 / g2_ol2l 1.022-1.087 on 8192
# and the eight P_K-16 band rows -- production and the "production at 32768 /
# 65536" leave the sm_103a table. The ns96 family (n_stack="all", incl. the early_tma forms) is a changed FP32
# grouping, gate-passing but not bitwise vs production, FAILS G1 T = 2433 cancel_delta dscale on 152 SMs
# and is out of every table; every early_tma form (fenced or not) is EXCLUDED from every table: racecheck
# does not complete on it (cfg 49: not completed in 3600 s at T = 129 while the knob-off forms complete in 5 s);
# st3ts (a_path="st", stages=3; exact) is no longer routed. sm_100a ADOPTS the same three-form rule
# (two valid sessions on two B200 nodes, vs production: nsx_st5_mb1 (45) 0.9454 / 0.9426 at 1024
# and 0.9495 / 0.9542 at 1152, nsx (39) 0.9829 / 0.9835 at 4096, nsx_g2_ol2l (54) 0.9292 / 0.9345 at 8192,
# 0.9032-0.9611 on the band rows 14928-16263, 0.8947 / 0.9014 at 32768, 0.9031 / 0.9094 at 65536 -- every row >= 1 %
# in both sessions; the 8191 boundary rests on 4096 -> 39 and 8192 -> 54 on both architectures). sm_107a ADOPTS the
# three-form rule in its own reading (two class-of-record sessions, bitwise vs production
# at the same P_K, controls 0.9962-1.0039): "nsx_st5_mb1" (47) where the K units fit the SM count (8 / 8 fit rows
# >= 1 % in both sessions, 0.9143-0.9535 / 0.9103-0.9608; 1024 0.9309 / 0.9311), "nsx_g2_ol2l" (56) on every other
# row up to 32768 tokens (4096 0.9451 / 0.9482 -- the 4096 -> nsx reading of sm_100a / sm_103a does not transfer:
# nsx reads 0.9911 / 0.9935 there --, 8192 0.9600 / 0.9672, the band rows 0.9293-0.9861 / 0.9432-0.9881, 32768
# 0.9723 / 0.9589) and cake_r1's production "ss_st4" above 32768 (65536: 56 reads 0.9911 / 0.9710 and nsx 0.9946 /
# 0.9954 -> no adoption, the threshold); 12 changed rows (every T of record but 65536).
FWD_MAP_PRODUCTION = "ss_st4"
FWD_MAP_BANDS = {
    # sm_100a routes the wider-N stacked form "ns96_g2_ol2l" (CONFIGS 57: [b1; b2; b3] as ONE N96 operand, 4 MMAs per stage,
    # ld_group 2, out_l2 evict_last; gate-passing, not bitwise: the FP32 grouping of z changes; G1-G7 / G11 and the T 2433 seed
    # sets PASS on sm_100a) on every row to 8192 tokens whose K units do not fit the SM count; where they fit, FWD_MAP_FIT keeps
    # "nsx_st5_mb1" (1024 / 1152 and the P_K-8 fit segment, where the stacked form reads 1.04-1.06 of it). It reads 0.93-0.97 of
    # "nsx" on the non-fit rows below 8192 and 0.95 of "nsx_g2_ol2l" at 8192 on B200; above 8192 the exact "nsx_g2_ol2l" form (on
    # the 14928-16263 rows the wider-N form's gain is inside the pairing spread of the checkpoint recompute, so the package path
    # rows do not qualify it there). sm_103a keeps the exact forms (its seed-set
    # gate fails); sm_107a gains < 1 % and is not routed.
    "sm_100a": ((0, 8192, "ns96_g2_ol2l"), (8192, None, "nsx_g2_ol2l")),
    "sm_103a": ((0, 8191, "nsx"), (8191, None, "nsx_g2_ol2l")),
    "sm_107a": ((0, 32768, "nsx_g2_ol2l"),),
}
# FWD_MAP_FIT: (program, replacement) per architecture -- where the band names `program` and the K units of the call
# fit the SM count (k1_units(T, sm_count, arch).units <= sm_count: every CTA runs alone on its SM), `replacement` is
# launched instead. On sm_103a: "nsx_st5_mb1" (stages=5, min_blocks=1: the 5-stage ring at one CTA per SM
# on the exact stacked form, 5 x 28 KiB + control; racecheck completes: 129 / 2049 / 16263 in 5.3 / 9.0 / 33.7 s;
# >= 1 % faster than nsx at EVERY fit-regime T in both sessions: nsx / nsx_st5_mb1 1.0299 / 1.0362 at 1024, 1.0289 /
# 1.0323 at 1152, 1.0109 / 1.0122 at 1793, 1.0371 / 1.0309 at 2049, 1.0307 / 1.0348 at 2432; vs production 0.9357 /
# 0.9310 at 1024) replaces "nsx" where the units fit: T <= 1152 (P_K 16 / 32) and the cost model's P_K = 8 segment
# 1793..2432 on 152 SMs. The same pair on sm_100a (45 vs production 0.9454 / 0.9426 at 1024 and 0.9495 /
# 0.9542 at 1152 on 148 SMs, evaluated at the cost model's P_K of that board). On sm_107a the pair is
# ("nsx_g2_ol2l", "nsx_st5_mb1") inside its one band (every fit T on 208 SMs lies inside the band; the P_K
# rule stays cake_r1's there).
FWD_MAP_FIT = {
    "sm_100a": (
        "ns96_g2_ol2l",
        "nsx_st5_mb1",
    ),  # the fit replacement follows the band program of the non-fit rows
    "sm_103a": ("nsx", "nsx_st5_mb1"),
    "sm_107a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
}
# FWD_MAP_CLAIM: (program, ticket form, tail form) per architecture -- the schedule of the routed program by the K units
# per CTA, u = units / grid of k1_units. The ticket form (schedule="ticket" with the control records at the tail of
# the ring; bitwise the program) claims each next K unit from the self-resetting ticket word, so when a concurrent
# kernel holds SMs the CTAs that start late take only the units left. It runs where u >= 4: unloaded 0.9937-1.0031 of
# the program in two sessions on two nodes per board (GB300, GB200, B200), and the forward path at 1.003-1.073 x its
# unloaded time with SMs held by a side kernel or an NCCL replica, where the program reads 1.158-1.291 x. Where
# 3 <= u < 4 the static schedule already gives every CTA 3 or 4 units and the claim costs up to 1.8 %; where u <= 1
# every CTA owns one unit and the claim balances nothing (1.0057-1.0102). There the program runs, or at u <= 1 its
# tail form where one exists (the static schedule with the control records at the tail, bitwise; sm_103a T 4096:
# 1.0011 / 0.9988 of the program, 1.11 x vs 1.29 x its unloaded time with one side CTA per SM; sm_100a
# ns96_g2_ol2l_ct, T 4096 on 148 SMs: 1.0000 / 0.9993 of the program, the mapping pass 1.01 x vs 1.43 x its
# unloaded time with one side CTA per SM; nsx_g2_ol2l has no tail form: no sm_100a row with u <= 1 was measured
# on it). sm_107a has no 216-SM timing session and is not routed.
FWD_MAP_CLAIM = {
    "sm_100a": (
        ("ns96_g2_ol2l", "tk_ns96_g2_ol2l_ct", "ns96_g2_ol2l_ct"),
        ("nsx_g2_ol2l", "tk_nsx_g2_ol2l_ct", None),
    ),
    "sm_103a": (
        ("nsx", "tk_nsx_ct", "nsx_ct"),
        ("nsx_g2_ol2l", "tk_nsx_g2_ol2l_ct", None),
    ),
    "sm_107a": (),
}
# FWD_COEF_REGIMES: the coefficient pass's dispatch per architecture, in grid units of the 8-token CTA (grid =
# ceil(T / 8) = mhc_det_fwd_coef.grid_rule(T, tokens_per_cta=8); every form bitwise the production chain). "round5" =
# the `form_overrides(T, sm_count, P_K)` of record on sm_103a (P_K = the host rule's K-unit
# count k1_units(T, sm_count, arch).P_K incl. PK_BANDS): "st16_w4_tree" (tokens_per_cta=4, stage=16, sum_tree;
# 0.9424-0.9601 of st16_w4 at 1024-1208) while grid < sm_count and P_K <= 16, "qe_mb10_tree" (tokens_per_cta=8,
# lane_pack=2, pair="adjacent", min_blocks=10, q_early, sum_tree) there at P_K > 16 (1153-1208 on 152 SMs: the
# 16-plane slab does not hold P_K = 32) and while 2 grid <= 9 sm_count (0.9367-0.9812 of qe_mb10 at 1216-5120);
# "lp2_w8_adj" (register fold) beyond 32768 tokens; at P_K > 8 (the P_K = 16 band of sm_103a plus the cost model's
# P_K = 16 rows below it) "st16_lp2_as_el_qd" (tokens_per_cta=8, lane_pack=2, pair="adjacent", stage=16, stage_async,
# stage_l2="evict_last", q_direct: the q partials loaded per lane straight from global memory before the slab copies;
# 0.9595 / 0.9598 of st16_lp2_as_el at 8192, 0.9663-0.9912 of qe_mb10_tree at 16114-16263, bitwise 861 / 861,
# sanitizer 9 / 9) while 2 grid <= 27 sm_count (5473..16416 on 152 SMs: one merged clause replacing round 4's
# st16_lp2_as_el up to 13 sm_count and qe_mb10_tree up to 2 grid <= 27 sm_count) and "qe_mb10" above (unmeasured at
# P_K > 8 there: the round-3 form); at P_K <= 8 the round-3 table: "qe_mb10" while grid <= 10 sm_count, "qe_mb10_pu4"
# (plane_unroll=4) while grid <= 12 sm_count, "st8_lp2" (stage=8) up to 32768 tokens. "round4" = the same rule with
# the round-4 P_K > 8 clauses -- "st16_lp2_as_el" (the hinted one-wave cp.async slab) while grid <= 13 sm_count,
# "qe_mb10_tree" while 2 grid <= 27 sm_count, "qe_mb10" above -- which is sm_107a's rule of record (FINAL for the rows
# the class-of-record second sessions cover: the round-3 dispatch and the round-4
# forms st16_w4_tree / qe_mb10_tree / st16_lp2_as_el; evaluated with cake_r1's P_K on 208 / 216 SMs; the round-5
# q_direct form is unpaired there). "round7" = sm_100a's rule of record (the round-3 dispatch
# at the cost model's P_K that the pairs kept on B200 (KEEP 13 / 13, Gate R max 0.8399), with the GB300 round-4 /
# round-5 forms A10 beta read ARCH-SPECIFIC CANDIDATE on B200 in two valid sessions adopted as existing CONFIGS
# entries): "st16_w4_tree" while grid < sm_count (1024: 0.9469 / 0.9457 of st16_w4), in the qe_mb10 regime (grid <=
# 10 sm_count) "qe_mb10_tree" (4096: 0.9642 / 0.9704 of qe_mb10) at P_K <= 8 and "st16_lp2_as_el_qd" at P_K = 16 (8192
# at the cost model's P_K 16 on 148 SMs: 0.9605 / 0.9637 of qe_mb10; the P_K = 32 rows 1177-1208 keep the tree form:
# the 16-plane slab does not hold P_K = 32), "qe_mb10_pu4" while grid <= 12 sm_count (the band 11841-14208 holds no
# record row on 148 SMs: information), in the band regime 12 sm_count < grid <= 14 sm_count (the rows
# 14928-16263 on 148 SMs; 14 = the smallest integer CTAs-per-SM bound holding the eight rows there, 16576 tokens:
# the reading of the "band regime", which names the rows and no bound) "st8_lp2_as_el_qd" (st8_lp2 +
# stage_async + stage_l2 evict_last + q_direct; a_el / as_qd 0.9770-0.9976 on the eight band
# rows in both sessions, never slower, >= 1 % in both at 15590 / 16164, 0.955-0.976 of production on every row,
# bitwise; CONFIGS 39 = the bytes of the `a_el` arm) where the rule's P_K is 8, "st8_lp2"
# beyond it up to 32768 tokens (as_qd 1.0025 / 1.0022 at 32768 = no gain; qe_mb10 where the P_K is 16: never
# on sm_100a, whose P_K rule is the cost model's), "lp2_w8_adj" beyond.
# "round3" = that dispatch without
# the adoptions (the sm_100a rule of rounds 3-6): "st16_w4" while grid < sm_count, "qe_mb10" while grid <= 10
# sm_count, "qe_mb10_pu4" while grid <= 12 sm_count, "st8_lp2" up to 32768 tokens at P_K 8 (qe_mb10 at 16),
# "lp2_w8_adj" beyond. "round2" = the round-2 dispatch (superseded on sm_100a): "st16_w4"
# while T < 4096; "mb10_lp2" while grid <= 12 sm_count; "st8_lp2" up to 32768; "lp2_w8_adj" beyond. "production" =
# the 16-token production chain "w16" at every T (no architecture). "round9" = sm_107a's rule of record
# (two class-of-record sessions on 208 SMs, bitwise vs production, valid controls; bands by measured T): "st16_w4_tree" (the form that closed the 1024 row) up to
# 1024 tokens, "pfl2_qmt" (qe_mb10_tree + prefetch_l2; ON qe_mb10_tree 0.9496 / 0.9346 at 4096) up to 4096,
# "st16_lp2_as_el_qd" (st16_lp2_as_el + q_direct -- qd ON st16_lp2_as_el 0.9598 / 0.9615 at 8192 in two
# sessions, bitwise, valid controls; CONFIGS 38 on both branches; st16_lp2_as_el 35 = information) up to
# 8192, "g2_el" (st16_lp2_as_el + groups_per_cta 2;
# ON st16_lp2_as_el 0.9337 / 0.9371 at 14928) up to 14928, "qe_mb10_tree" (ON qe_mb10 0.9550-0.9638 on 7 / 7 rows
# 15288-16263) up to 16263, "st8_lp2_tree_asel_qd" (st8_lp2 + sum_tree + stage_async + stage_l2 evict_last + q_direct
# -- qd ON st8_lp2_tree_asel 0.9689 / 0.9690 at 32768 in two sessions; before it asel ON st8_lp2_tree
# 0.9869 / 0.9897, tree ON st8_lp2 0.9764 / 0.9750 there) up to 32768, "lp2_w8_adj_qe"
# (lp2_w8_adj + q_early; ON lp2_w8_adj 0.9100 / 0.9178 at 65536; qe_tree = information: no same-process pair,
# the issue comment on the K1 65536 pair) beyond. The round-4 rule's P_K and CTA-per-SM clauses do not survive: every band
# holds its form at every P_K of the cost model, on 208 and 216 SMs alike (CONFIGS 26 / 42 / 38 / 43 / 29 / 48 /
# 45; st8_lp2_tree_asel 47 and st16_lp2_as_el 35 = information).
FWD_COEF_REGIMES = {"sm_100a": "round7", "sm_103a": "round5", "sm_107a": "round9"}
# The backward programs of record per architecture: K6 "paring" (production) / "dot_ao8" (ring_dot=True, act_offset=8;
# KEEP 13 / 13 on B200 and on R200, FINAL) with the per-R regimes of BWD_TOKEN_REGIMES -- the deterministic package's
# token-owner rules, verbatim.  K7 is this package's column slab: the dynr record
# (``cake_mhc_train_bwd_slab_atomic__dynr``, the deterministic "dynr" form with FP32-reduced dfn and the parameter
# fold) on every architecture, or its two-size tail schedule ``lpt`` where the screen of BWD_SLAB_LPT (H 4096) or
# BWD_SLAB_LPT_HIDDEN (H 5120 / 7168) fires (``bwd_slab_atomic_program``): the deterministic package's
# per-architecture column-slab and final-reduction (K8) rules do not apply here.
# BWD_TOKEN_PROGRAM: the token-owner program per architecture outside every regime of BWD_TOKEN_REGIMES and every band
# of BWD_TOKEN_BANDS. sm_107a: "dot_ao8_l2l" (dot_ao8 +
# ring_l2=evict_last; CONFIGS 23) -- vs dot_ao8 s1 0.9506..0.9804 / s2 0.9531..0.9750 on the ten rows
# 14928-65536, two class-of-record sessions on 208 SMs, bitwise, valid controls. On 208 / 216 SMs every record row
# above 8192 lies outside the R regimes (14928-16263 -> R 40; 32768 / 65536 -> R >= 80), so the program is the form of
# those rows; the regimes below keep F_A (R <= 32) and the TMA-ring F_B (48 < R <= 56: no record row).
BWD_TOKEN_PROGRAM = {
    "sm_100a": "dot_ao8",
    "sm_103a": "dot_ao8",
    "sm_107a": "dot_ao8_l2l",
}
# BWD_TOKEN_REGIMES: ((program, R_min, R_max), ...) per architecture over R = k6_grid(T, sm_count).R, the tokens per
# CTA (a multiple of 8; sm_count-aware, never a baked T constant): F_A "ao22_rae7_pd1_ntr" (ring_dot=True,
# ring_act_every=7, act_offset=22, pro_defer=1, tail_refill=False; 0.958-0.984 of dot_ao8 at 1024 / 4096 / 8192 on
# GB300; KEEP in its regime on B200 and on R200, FINAL there) for R <= 32 on every architecture; for 48 <
# R <= 56 the TMA-ring form "ao15_rae8_tma_ntr" (F_B "ao15_rae8_ntr" = ring_dot=True, ring_act_every=8, act_offset=15,
# tail_refill=False -- 0.993-0.998 of dot_ao8 at 14928-16263, KEEP 8 / 8 on B200 -- plus ring_tma=True: X / dU as
# tensor-map descriptors) on every architecture: on sm_103a (0.9832-0.9926 of F_B on the eight regime-B rows
# in two sessions, bitwise, sanitizer 9 / 9); on sm_107a FINAL (regime B holds no record row on 208 / 216
# SMs: T in 19969..23296 / 20737..24192); on sm_100a ADOPTED (two valid sessions: TMA-B / F_B
# 0.9805-0.9897 on the eight rows 14928-16263, 8 / 8 >= 1 % in both; the A10 s2 PM-error session is replaced; F_B
# leaves the sm_100a table); BWD_TOKEN_PROGRAM elsewhere (R 40 / 48 and R >= 64: F_A +2.9..+3.5 % at R 40 / 48,
# +0.9..+1.7 % at 16114-65536; F_B +0.1..+0.4 % there). Lane k6 spells the symbols `_dot_ao22_rae7_pd1_ntr` /
# `_dot_ao15_rae8_ntr` / `_dot_ao15_rae8_tma_ntr` (CONFIGS 7-9 / 10-12 / 13-15, 1-based).
BWD_TOKEN_REGIMES = {
    "sm_100a": (("ao22_rae7_pd1_ntr", 0, 32), ("ao15_rae8_tma_ntr", 49, 56)),
    "sm_103a": (("ao22_rae7_pd1_ntr", 0, 32), ("ao15_rae8_tma_ntr", 49, 56)),
    "sm_107a": (("ao22_rae7_pd1_ntr", 0, 32), ("ao15_rae8_tma_ntr", 49, 56)),
}
# BWD_TOKEN_BANDS: (above, at_most, program) by token count per architecture, consulted before the R regimes
# (at_most None = unbounded): on sm_107a the F_A derivative "ao22_rae7_rf3_pd1_l2l_ntr" (F_A + ring_l2=evict_last
# + refill_sub=3; CONFIGS 21) for T <= 1024 -- the adoption-pair winner ON F_A, 0.9730 / 0.9685
# at 1024 in two class-of-record sessions, bitwise, valid controls; 1.0034 / 1.0039 at 4096 -> 4096 stays F_A. A
# token-count band because k6_grid routes R = 8 for every T <= 16 x SMs: the R regimes cannot single 1024 out.
# "ao22_rae7_pd1_tma_ntr" (F_A + ring_tma; CONFIGS 22) for 1024 < T <=
# 4096 -- F_A + ring_tma vs F_A 0.9838 / 0.9869 at 4096 in two class-of-record sessions, bitwise, valid controls; 8192
# stays F_A (the R regimes): the pairs ON F_A at 8192 read ring_l2=evict_last 0.9904 / 0.9930,
# fold_sync=token 0.9901 / 0.9835, both 0.9944 / 0.9911 (not >= 1 % in both sessions); those candidates stay
# CONFIGS entries routed nowhere (24 / 25 / 26).
# sm_100a: "ao22_rae7_pd1_tma_ntr" (F_A with the TMA-filled ring; the sm_100a CONFIGS entry appended with this band) for
# 1024 < T <= 8192 -- the token-count band covers the F_A regime's rows of record 4096 / 8192 (R <= 32; at 1024 the ring-filled
# form is inside the pairing spread, so F_A keeps that row; 8193..9472 stay
# F_A: no row of record there). The band rows 14928-16263 keep TMA-B, which the TMA-filled F_A does not beat there.
BWD_TOKEN_BANDS = {
    "sm_100a": ((1024, 8192, "ao22_rae7_pd1_tma_ntr"),),
    "sm_103a": (),
    "sm_107a": (
        (0, 1024, "ao22_rae7_rf3_pd1_l2l_ntr"),
        (1024, 4096, "ao22_rae7_pd1_tma_ntr"),
    ),
}
# K5_GRID_BANDS: (above, at_most, grid_per_sm) per architecture -- the backward prepare's launch grid by token count
# inside a band the design's persistent M1 grid min(ceil(T / 2), grid_per_sm x sm_count) (grid-stride
# trips of the same binary: a host rule only, the program bytes are unchanged), outside every band the one-trip
# production grid ceil(T / 2) (mhc_det_bwd_prepare.grid_rule(T, sm_count, rows_per_cta=2)). On sm_103a 2 x SMs = 304
# CTAs for T <= 4096: g304 / production 0.9879 / 0.9865 / 0.9888 / 0.9865 at 1024 and 0.9868 / 0.9884 / 0.9872 /
# 0.9863 at 4096 in four sessions on two nodes (bitwise, repeat 0; sanitizer PASS at grids 256 / 304 in both g2
# sessions; g256 no gain at 4096, g456 a regression), slower on all eleven rows above 4096 in both sessions
# (1.0067-1.0204) -> the band ends at 4096. On sm_100a 2 x SMs = 296 CTAs for T <= 1024 (g296
# / production 0.9863 / 0.9755 at 1024, controls 1.0018 / 1.0013, bitwise, repeat 0, sanitizer 9 / 9 PASS at grid 296;
# 4096 0.9889 / 0.9926 = no adoption; above 4096 <= 0.99 in 0 / 11 rows -> production) -> the band ends at
# 1024. sm_107a keeps the production grid (the K5 grid verdict of record
# reads g416 = 2 x 208 vs production <= 0.99 in neither session -- s1 1024 0.9902 / 4096 0.9934, s2
# 1024 0.9953 / 4096 1.0093, sanitizer 9 / 9 in both -> no band, K5 retained).
K5_GRID_BANDS = {"sm_100a": ((0, 1024, 2),), "sm_103a": ((0, 4096, 2),), "sm_107a": ()}
# K5_FORM_BANDS: (above, at_most, form) per architecture -- the backward prepare's CTA form by token count, read before
# K5_GRID_BANDS: "t128" = the production knobs on 128-thread CTAs of one token (the CONFIGS entries with threads 128,
# rows_per_cta 1, min_blocks 4: the record's 128-thread token group, so the per-token tree and the bits are the
# record's) on the one-trip grid T; outside every band the 256-thread two-token record ("production": ceil(T / 2) CTAs,
# or the persistent grid of a K5_GRID_BANDS band). On sm_100a the t128 form for 1024 < T <= 8192; on sm_103a the t128
# form to 8192 tokens; on sm_107a the t128 form to 65536 tokens (the form's one-trip grid is faster than the record's
# grid of record at the rows of record inside the band; the persistent 4 x SMs t128 grid ties the one-trip grid at 1024
# tokens and loses above it, so the form has no persistent band).
K5_FORM_BANDS = {
    "sm_100a": ((1024, 8192, "t128"),),
    "sm_103a": ((0, 8192, "t128"),),
    "sm_107a": ((0, 65536, "t128"),),
}
K5_FORM_ROWS = {
    "production": 2,
    "t128": 1,
}  # tokens per CTA trip of each form (its one-trip grid is ceil(T / rows))
# FWD_FINISH_PROGRAM: the residual mixing's program per architecture: "production"
# = cake_r1's binary (rows_per_cta 4, col_split 2, coef_first; the one-trip grid ceil(T / 2)); on sm_107a "r2" = the
# production knobs with min_blocks 2 and two rows per CTA (CONFIGS 7; grid T): the sets ON mb2 read r2
# <= 0.99 in both class-of-record sessions on 13 / 13 rows (0.9569-0.9843 / 0.9577-0.9872), mb2 itself the
# candidate of record ON production (12 / 13 rows 0.7874-0.8854). k4_grid(T) stays the production grid;
# k4_grid_of_record(T, arch) is the launch grid of the program of record (FWD_FINISH_ROWS: its rows per CTA, col_split
# 2 in both) and the rules table's k4_grid column.
FWD_FINISH_PROGRAM = {"sm_100a": "production", "sm_103a": "production", "sm_107a": "r2"}
FWD_FINISH_ROWS = {"production": 4, "r2": 2}
# BWD_SLAB_LPT: the column slab's two-size tail schedule per architecture -- None (the "dynr" record at every token
# count) or (o_tiles, keep): lpt_chunks ranks the two-size geometries of T tokens (big chunks of rb tiles, rb within
# six tiles of the one-size k7_chunks chunks and leaving tokens after them, then S = 1..5 small chunks sharing the
# rest; at most 32 chunks) by the list-scheduling makespan of their units in draw order on the device's SMs (a unit
# costs o_tiles plus its 64-token tiles, each to the least loaded SM, the big units first; the fewest chunks, then the
# biggest, among equal makespans) and runs the best on the "lpt" record where its makespan is at most keep times the
# one-size chunks'. Calibrated by package-level A/B: the rule's geometry against dynr and a null-control rerun of dynr
# in one process, paired per replay group, on the finish / backward / checkpointed-step graphs; the deltas below are
# against the faster dynr arm. sm_100a (0.5, 0.99): on 148 SMs it fires at
# 1536 / 2048 / 2560 / 3584 / 14928 / 15566 / 15590 / 16114 of the 19 benchmark token counts (finish
# -1.2 / -2.1 / -3.5 / -6.7 / -5.4 / -14.0 / -12.9 / -6.6 us), and over 67 firing token counts in 1024..65536 finish
# -6.4 us (-1.1 %) on average, the seven slower ones at most +2.0 % (22155, whose groups span -52..+137 us). sm_107a
# (0.5, 0.97): on 212 SMs it fires at 1024 / 2048 / 2560 / 3072 / 3584 of the benchmark token counts (finish
# -0.6 / -1.5 / -1.7 / -1.2 / -0.6 us) and at 21 of 160 token counts in 1024..65536, every one faster (finish -2.7 %,
# checkpointed step -1.2 % on average); keep 0.98 adds token counts that read up to +15 us (34721..37194, 53378,
# 61944..65181). sm_103a (0.5, 0.98): on 152 SMs it fires at
# 1024 / 1536 / 2048 / 8192 / 14928 / 15288 / 16132 / 16164 / 16263 of the benchmark token counts (finish
# -0.7 / -0.9 / -1.9 / -3.7 / -0.8 / -7.4 / -5.8 / -6.0 / -2.2 us) and at 27 of 160 token counts in 1024..65536, none
# slower by more than 0.2 us (finish -1.1 %, checkpointed step -0.45 % on average); keep 0.985 adds a token count
# whose checkpointed step reads +0.6 us in every group (26216), keep 0.99 one whose finish reads +1.6 us (16444).
# Experiments that edit this table or BWD_SLAB_LPT_HIDDEN or replace lpt_chunks clear the package's _PLANS /
# _FAST_ROUTES: the selection and slab_plan read them at call time, the route plans bind them per token count, and
# host_plan carries none of them. The table is calibrated at H 4096; lpt_chunks reads BWD_SLAB_LPT_HIDDEN at the other
# sizes.
BWD_SLAB_LPT = {"sm_100a": (0.5, 0.99), "sm_103a": (0.5, 0.98), "sm_107a": (0.5, 0.97)}
# BWD_SLAB_LPT_HIDDEN: BWD_SLAB_LPT per hidden size other than 4096 -- the same screen over the size's H / 64 slabs and
# its k7_chunks chunks, calibrated per (architecture, hidden size); a size it does not list is a ValueError, never the
# H 4096 value. None where no lpt__h<H> record is routed: sm_103a (no GB300 calibration of these sizes yet) and sm_107a
# (hidden 4096 only). sm_100a, by the package-level A/B of BWD_SLAB_LPT on 148 SMs over 80 token counts in 1024..65536
# (the ones that fire at keep 0.995, 8 replay groups): H 5120 (1.0, 0.99) fires at 28 of them, backward -188.5 us and
# checkpointed step -113.4 us in sum, none slower in a 16-group rerun (20656 reads +0.1 / -0.1 us); keep 0.995 fires at
# 46 but adds 25534 and 34557, whose backward reads +7.6 / +12.9 us in every one of 16 groups. H 7168 (1.0, 0.98) fires
# at 6, backward -38.6 us and checkpointed step -21.3 us in sum, none slower; keep 0.99 / 0.995 are slower in sum
# (backward +1.0 / +93.7 us). Both losing sets are geometries whose makespan gain is under 1.5 % or whose big chunks
# outnumber the one-size chunks. Of the 19 benchmark token counts it fires at H 5120 T 1024 / 8192 / 15566 / 15590 /
# 16263 / 32768 (backward -1.3 / -2.9 / -9.6 / -11.4 / -11.7 / -3.1 us, 16 replay groups) and H 7168 T 1024 / 2048 /
# 3072 / 15288 (-3.8 / -4.8 / -4.8 / -10.1 us), and at H 7168 also at 512 (-1.8 us) and 24576 (+2.2 us within its 4 us
# null, checkpointed step -21 us).
BWD_SLAB_LPT_HIDDEN = {
    5120: {"sm_100a": (1.0, 0.99), "sm_103a": None, "sm_107a": None},
    7168: {"sm_100a": (1.0, 0.98), "sm_103a": None, "sm_107a": None},
}

# FWD_FINISH_COL_SPLIT: the residual mixing's sub-tokens per token (half tokens of 2048 columns at H 4096).
FWD_FINISH_COL_SPLIT = 2
# K3_TILING: the collapse's (rows_per_cta, col_split) of record (k3_grid): one half token per CTA at H 4096.
K3_TILING = (1, 2)
# K6_CTAS_PER_SM: the token owner's resident CTAs per SM, the factor of k6_grid's G6 (two 90112-B rings at H 4096).
K6_CTAS_PER_SM = 2

# H_TABLES: the tables of each hidden size other than 4096, read through ``_tab(name, hidden)`` as in the
# deterministic registry (the tables above are those of record at H 4096). H_TABLE_NAMES is the deterministic list
# without the column-slab and final-reduction tables, and H_TABLES[H] is the deterministic H_TABLES[H] of those names
# for H 5120 and 7168 (the forms of record of those sizes and their rationale are the deterministic registry's); a size
# it does not spell is an error, never the H 4096 value. The column slab's calibration of those sizes is no H_TABLES
# entry: lpt_chunks reads BWD_SLAB_LPT_HIDDEN (BWD_SLAB_LPT is the H 4096 table).
H_TABLE_NAMES = (
    "PK_BANDS", "FN_SPLIT_PROGRAM", "FN_SPLIT_GRID", "FWD_MAP_PRODUCTION", "FWD_MAP_BANDS", "FWD_MAP_FIT",
    "FWD_MAP_CLAIM", "FWD_COEF_REGIMES", "BWD_TOKEN_PROGRAM", "BWD_TOKEN_REGIMES", "BWD_TOKEN_BANDS",
    "K5_GRID_BANDS", "K5_FORM_BANDS", "K5_FORM_ROWS", "FWD_FINISH_PROGRAM", "FWD_FINISH_ROWS",
    "FWD_FINISH_COL_SPLIT", "K3_TILING", "K6_CTAS_PER_SM",
)  # fmt: skip
H_TABLES: dict[int, dict[str, Any]] = {
    5120: {
        "PK_BANDS": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "FN_SPLIT_PROGRAM": {
            "sm_100a": ("b16", "tv128"),
            "sm_103a": ("b16", "tv128"),
            "sm_107a": ("b16", "tv128_t128"),
        },
        "FN_SPLIT_GRID": {"b16": 320, "tv128": 320, "tv128_t128": 640},
        "FWD_MAP_PRODUCTION": "nsx_g2_ol2l",
        "FWD_MAP_BANDS": {
            "sm_100a": ((0, None, "nsx_g2_ol2l"),),
            "sm_103a": ((0, None, "nsx_g2_ol2l"),),
            "sm_107a": ((0, None, "nsx_g2_ol2l"),),
        },
        "FWD_MAP_FIT": {
            "sm_100a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
            "sm_103a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
            "sm_107a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
        },
        "FWD_MAP_CLAIM": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "FWD_COEF_REGIMES": {
            "sm_100a": "round7",
            "sm_103a": "round5",
            "sm_107a": "round9",
        },
        "BWD_TOKEN_PROGRAM": {
            "sm_100a": "dot_ao8",
            "sm_103a": "dot_ao8",
            "sm_107a": "dot_ao8_l2l",
        },
        "BWD_TOKEN_REGIMES": {
            "sm_100a": (("ao22_rae7_pd1_ntr", 0, 32),),
            "sm_103a": (("ao22_rae7_pd1_ntr", 0, 32),),
            "sm_107a": (("ao22_rae7_pd1_ntr", 0, 32),),
        },
        "BWD_TOKEN_BANDS": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "K5_GRID_BANDS": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "K5_FORM_BANDS": {
            "sm_100a": ((0, None, "t160"),),
            "sm_103a": ((0, None, "t160"),),
            "sm_107a": ((0, 8192, "t160"),),
        },
        "K5_FORM_ROWS": {"production": 2, "t160": 1},
        "FWD_FINISH_PROGRAM": {"sm_100a": "w2", "sm_103a": "w2", "sm_107a": "w1"},
        "FWD_FINISH_ROWS": {"w2": 8, "w1": 4},
        "FWD_FINISH_COL_SPLIT": 5,
        "K3_TILING": (2, 5),
        "K6_CTAS_PER_SM": 2,
    },
    7168: {
        "PK_BANDS": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "FN_SPLIT_PROGRAM": {
            "sm_100a": ("b16", "tv128"),
            "sm_103a": ("b16", "tv128"),
            "sm_107a": ("b16", "tv128_t128"),
        },
        "FN_SPLIT_GRID": {"b16": 448, "tv128": 448, "tv128_t128": 896},
        "FWD_MAP_PRODUCTION": "nsx_g2_ol2l",
        "FWD_MAP_BANDS": {
            "sm_100a": ((0, None, "nsx_g2_ol2l"),),
            "sm_103a": ((0, None, "nsx_g2_ol2l"),),
            "sm_107a": ((0, None, "nsx_g2_ol2l"),),
        },
        "FWD_MAP_FIT": {
            "sm_100a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
            "sm_103a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
            "sm_107a": ("nsx_g2_ol2l", "nsx_st5_mb1"),
        },
        "FWD_MAP_CLAIM": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "FWD_COEF_REGIMES": {
            "sm_100a": "round7",
            "sm_103a": "round5",
            "sm_107a": "round9",
        },
        "BWD_TOKEN_PROGRAM": {
            "sm_100a": "dot_ao8",
            "sm_103a": "dot_ao8",
            "sm_107a": "dot_ao8_l2l",
        },
        "BWD_TOKEN_REGIMES": {
            "sm_100a": (("ao22_rae7_pd1_ntr", 0, 32),),
            "sm_103a": (("ao22_rae7_pd1_ntr", 0, 32),),
            "sm_107a": (("ao22_rae7_pd1_ntr", 0, 32),),
        },
        "BWD_TOKEN_BANDS": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "K5_GRID_BANDS": {"sm_100a": (), "sm_103a": (), "sm_107a": ()},
        "K5_FORM_BANDS": {
            "sm_100a": ((0, None, "t224"),),
            "sm_103a": ((0, None, "t224"),),
            "sm_107a": ((0, 8192, "t224"),),
        },
        "K5_FORM_ROWS": {"production": 2, "t224": 1},
        "FWD_FINISH_PROGRAM": {"sm_100a": "w2", "sm_103a": "w2", "sm_107a": "w1"},
        "FWD_FINISH_ROWS": {"w2": 8, "w1": 4},
        "FWD_FINISH_COL_SPLIT": 7,
        "K3_TILING": (2, 7),
        "K6_CTAS_PER_SM": 1,
    },
}


def _tab(name: str, hidden: int) -> Any:
    """The table ``name`` of hidden size ``hidden``: the module-level table of
    record at H 4096, ``H_TABLES[hidden]`` otherwise (``KeyError`` for a size or a
    table it does not spell: never the H 4096 value)."""
    if name not in H_TABLE_NAMES:
        raise KeyError(f"{name} is not a per-hidden-size table")
    if hidden == HIDDEN:
        return globals()[name]
    return H_TABLES[hidden][name]


def k1_units(T, sm_count, arch, hidden=HIDDEN):  # -> (P_K, units, grid)
    mb = -(-T // 128)
    best = None
    for pk in (8, 16, 32):
        units = mb * pk
        rounds = -(-units // sm_count)
        grid = min(
            2 * sm_count, units
        )  # SM-level rounds: both CTAs of an SM share its bandwidth and the ticket queue
        cost = (
            rounds * sm_count * 128 * (STREAMS * hidden // pk + 100) + 100 * pk * T
        )  # bytes-equivalent makespan + K2 partial reads
        if best is None or (cost, pk) < best[0]:
            best = ((cost, pk), pk, units, grid)
    pk = best[1]
    for above, at_most, band_pk in _tab("PK_BANDS", hidden)[arch]:
        if above < T <= at_most:
            pk = band_pk
    units = mb * pk
    return pk, units, min(2 * sm_count, units)


def fn_split_program(
    arch, want_t, hidden=HIDDEN
):  # -> the route's program (a FN_SPLIT_GRID key); want_t 0 fnp, 1 fnTp
    return _tab("FN_SPLIT_PROGRAM", hidden)[arch][want_t]


def fn_split_grid(
    arch, want_t, hidden=HIDDEN
):  # -> the route's launch grid (H / 16; H / 8 for tv128_t128)
    return _tab("FN_SPLIT_GRID", hidden)[fn_split_program(arch, want_t, hidden)]


def fwd_map_program(
    T, sm_count, arch, hidden=HIDDEN
):  # -> the mapping-pass program of record for T tokens
    program = _tab("FWD_MAP_PRODUCTION", hidden)
    for above, at_most, band in _tab("FWD_MAP_BANDS", hidden)[arch]:
        if above < T and (at_most is None or at_most >= T):
            program = band
    _, units, grid = k1_units(T, sm_count, arch, hidden)
    fit = _tab("FWD_MAP_FIT", hidden)[arch]
    if fit is not None and program == fit[0] and units <= sm_count:
        program = fit[1]
    for base, ticket, tail in _tab("FWD_MAP_CLAIM", hidden)[
        arch
    ]:  # the schedule by units per CTA (u = units / grid)
        if program == base:
            if ticket is not None and units >= 4 * grid:
                program = ticket
            elif tail is not None and units <= grid:
                program = tail
    return program


def fwd_coef_program(
    T, sm_count, arch, hidden=HIDDEN
):  # -> the coefficient-pass program of record for T tokens
    regime = _tab("FWD_COEF_REGIMES", hidden)[arch]
    if regime == "production":
        return "w16"
    grid = -(-T // 8)  # mhc_det_fwd_coef.grid_rule(T, tokens_per_cta=8)
    if regime == "round2":
        if T < 4096:
            return "st16_w4"
        if grid <= 12 * sm_count:
            return "mb10_lp2"
        return "st8_lp2" if T <= 32768 else "lp2_w8_adj"
    P_K = k1_units(T, sm_count, arch, hidden)[
        0
    ]  # the host rule's K-unit count of the launch (PK_BANDS included)
    if regime in (
        "round4",
        "round5",
    ):  # the form_overrides(T, sm_count, P_K) of record
        if grid < sm_count:
            return "qe_mb10_tree" if P_K > 16 else "st16_w4_tree"
        if 2 * grid <= 9 * sm_count:
            return "qe_mb10_tree"
        if T > 32768:
            return "lp2_w8_adj"
        if P_K > 8:
            if regime == "round5":  # one merged clause, the q_direct cp.async slab
                return "st16_lp2_as_el_qd" if 2 * grid <= 27 * sm_count else "qe_mb10"
            if grid <= 13 * sm_count:
                return "st16_lp2_as_el"
            return "qe_mb10_tree" if 2 * grid <= 27 * sm_count else "qe_mb10"
        if grid <= 10 * sm_count:
            return "qe_mb10"
        return "qe_mb10_pu4" if grid <= 12 * sm_count else "st8_lp2"
    if regime == "round9":  # sm_107a's forms of record by measured-T band
        if T <= 1024:
            return "st16_w4_tree"
        if T <= 4096:
            return "pfl2_qmt"
        if T <= 8192:
            return "st16_lp2_as_el_qd"
        if T <= 14928:
            return "g2_el"
        if T <= 16263:
            return "qe_mb10_tree"
        if T <= 32768:
            return "st8_lp2_tree_asel_qd"
        return "lp2_w8_adj_qe"
    if (
        regime == "round7"
    ):  # sm_100a's round-3 regimes with its adoptions (tree forms, q_direct slab)
        if grid < sm_count:
            return "st16_w4_tree"
        if grid <= 10 * sm_count:
            return "st16_lp2_as_el_qd" if 8 < P_K <= 16 else "qe_mb10_tree"
        if grid <= 12 * sm_count:
            return "qe_mb10_pu4"
        if grid <= 14 * sm_count:  # the band regime (the rows 14928-16263 on 148 SMs)
            return "st8_lp2_as_el_qd" if P_K <= 8 else "qe_mb10"
        if T <= 32768:
            return "st8_lp2" if P_K <= 8 else "qe_mb10"
        return "lp2_w8_adj"
    if grid < sm_count:
        return "st16_w4"
    if grid <= 10 * sm_count:
        return "qe_mb10"
    if grid <= 12 * sm_count:
        return "qe_mb10_pu4"
    if T <= 32768:
        return "st8_lp2" if P_K <= 8 else "qe_mb10"
    return "lp2_w8_adj"


def bwd_token_program(
    T, sm_count, arch, hidden=HIDDEN
):  # -> the token-owner pass's program of record for T tokens
    for above, at_most, program in _tab("BWD_TOKEN_BANDS", hidden)[
        arch
    ]:  # the token-count bands come first
        if above < T and (at_most is None or at_most >= T):
            return program
    R = k6_grid(T, sm_count, hidden)[1]
    for program, r_min, r_max in _tab("BWD_TOKEN_REGIMES", hidden)[arch]:
        if r_min <= R <= r_max:
            return program
    return _tab("BWD_TOKEN_PROGRAM", hidden)[arch]


def k5_program(
    T, sm_count, arch, hidden=HIDDEN
):  # -> the backward prepare's CTA form of record for T tokens
    for above, at_most, form in _tab("K5_FORM_BANDS", hidden)[arch]:
        if above < T and (at_most is None or at_most >= T):
            return form
    return "production"


def k5_grid(
    T, sm_count, arch, hidden=HIDDEN
):  # -> the backward prepare's launch grid for T tokens on sm_count SMs
    form = k5_program(T, sm_count, arch, hidden)
    grid = -(
        -T // _tab("K5_FORM_ROWS", hidden)[form]
    )  # the form's one-trip grid: grid_rule(T, sm_count, rows_per_cta=rows)
    if form == "production":  # the persistent bands belong to the record's form
        for above, at_most, grid_per_sm in _tab("K5_GRID_BANDS", hidden)[arch]:
            if above < T and (at_most is None or at_most >= T):
                grid = min(grid, grid_per_sm * sm_count)
    return grid


def fwd_finish_program(
    arch, hidden=HIDDEN
):  # -> the residual mixing's program of record per architecture
    return _tab("FWD_FINISH_PROGRAM", hidden)[arch]


def k4_grid_of_record(
    T, arch, hidden=HIDDEN
):  # -> the residual mixing's launch grid: ceil(col_split T / rows_per_cta)
    rows = _tab("FWD_FINISH_ROWS", hidden)[fwd_finish_program(arch, hidden)]
    return -(-(T * _tab("FWD_FINISH_COL_SPLIT", hidden)) // rows)


def k7_chunks(
    T, sm_count, slab_d=64, hidden=HIDDEN
):  # -> (P, units, grid, rows_per_chunk)
    S = STREAMS * hidden // (4 * slab_d)
    tiles = -(-T // 64)
    b_unit = 3 * 4 * slab_d * 2 + slab_d * 2 + 192 + 96
    best = None
    for p in range(1, min(32, tiles) + 1):
        rpc = 64 * (-(-tiles // p))
        units = S * p
        grid = min(sm_count, units)
        cost = (
            sm_count * (-(-units // sm_count)) * rpc * b_unit
            + 2 * p * 4 * MAPPING_ROWS * STREAMS * hidden
        )  # makespan bytes + partial write+read (1_572_864 B per chunk at H 4096)
        if best is None or (cost, p) < best[0]:
            best = ((cost, p), p, units, grid, rpc)
    return best[1:]


def _lpt_units(
    T, geometry, hidden=HIDDEN
):  # -> the units in draw order (every slab's big chunks, then the small ones) as [count, tiles] runs; H / 64 slabs
    P, rpc, p_big, rpc_small = geometry
    tiles, rb, rs, slabs = -(-T // 64), rpc // 64, rpc_small // 64, hidden // SLAB_D
    big = [max(0, min(rb, tiles - p * rb)) for p in range(p_big)]
    small = [max(0, min(rs, tiles - p_big * rb - q * rs)) for q in range(P - p_big)]
    runs = []
    for chunks in (big, small):
        pattern = []  # one slab's chunks as runs
        for t in chunks:
            if pattern and pattern[-1][1] == t:
                pattern[-1][0] += 1
            else:
                pattern.append([1, t])
        repeats = slabs
        if len(pattern) == 1:  # one tile count: every slab's units are one run
            pattern, repeats = [[pattern[0][0] * slabs, pattern[0][1]]], 1
        for _ in range(repeats):
            for count, t in pattern:
                if runs and runs[-1][1] == t:
                    runs[-1][0] += count
                else:
                    runs.append([count, t])
    return runs


def _lpt_makespan(
    runs, sm_count, o_tiles
):  # -> list scheduling: each unit to the least loaded SM; the largest load
    # The SMs of one load level take a run's units together: every SM's load is the same sum, in the same order, as
    # unit by unit, so the makespan is bit for bit the unit-by-unit one (the levels are few: the screen stays cheap).
    level = {0.0: min(sm_count, sum(count for count, _ in runs))}
    heap = [0.0]
    for count, tiles in runs:
        while count:
            load = heap[0]
            sms = level[load]
            moved = min(sms, count)
            if moved == sms:
                heapq.heappop(heap)
                del level[load]
            else:
                level[load] = sms - moved
            load = load + o_tiles + tiles
            if load in level:
                level[load] += moved
            else:
                level[load] = moved
                heapq.heappush(heap, load)
            count -= moved
    return max(level)


def _lpt_geometry(
    T, rb, S
):  # -> (P, rpc, p_big, rpc_small) of rb-tile big chunks and S small ones, or None
    tiles = -(-T // 64)
    if rb < 1 or S < 1:
        return None
    p_big = (tiles - 1) // rb
    rs = -(-(tiles - p_big * rb) // S)
    if p_big < 1 or p_big + S > 32:
        return None
    if S == 1 and rs == rb:
        return (p_big + 1, 64 * rb, p_big + 1, 64 * rb)
    if not 1 <= rs < rb:
        return None
    return (p_big + S, 64 * rb, p_big, 64 * rs)


@functools.lru_cache(maxsize=4096)
def _lpt_plan(
    T, sm_count, o_tiles, hidden=HIDDEN
):  # -> (one-size makespan, best makespan, best geometry) at hidden size hidden (its slabs and chunks), or None
    if T <= 0:
        return None
    P0, _, _, rpc0 = k7_chunks(T, sm_count, hidden=hidden)
    rt = rpc0 // 64
    baseline = _lpt_makespan(
        _lpt_units(T, (P0, rpc0, P0, rpc0), hidden=hidden), sm_count, o_tiles
    )
    best = None
    for rb in range(max(1, rt - 6), rt + 1):
        for S in range(1, 6):
            geometry = _lpt_geometry(T, rb, S)
            if geometry is None:
                continue
            key = (
                _lpt_makespan(
                    _lpt_units(T, geometry, hidden=hidden), sm_count, o_tiles
                ),
                geometry[0],
                -rb,
            )
            if best is None or key < best[0]:
                best = (key, geometry)
    return None if best is None else (baseline, best[0][0], best[1])


def lpt_chunks(
    T, sm_count, arch, hidden=HIDDEN
):  # -> the column slab's two-size geometry (P, rpc, p_big, rpc_small) at hidden size hidden, or None
    if hidden == HIDDEN:
        calibration = BWD_SLAB_LPT[arch]
    elif hidden in BWD_SLAB_LPT_HIDDEN:
        calibration = BWD_SLAB_LPT_HIDDEN[hidden][arch]
    else:
        raise ValueError(f"hidden must be one of {HIDDEN_SIZES}, got {hidden}")
    if calibration is None:
        return None
    o_tiles, keep = calibration
    plan = _lpt_plan(T, sm_count, float(o_tiles), hidden=hidden)
    if plan is None:
        return None
    baseline, makespan, geometry = plan
    return geometry if makespan <= keep * baseline else None


def bwd_slab_atomic_program(
    T, sm_count, arch, hidden=HIDDEN
):  # -> "lpt" where lpt_chunks gives a geometry, else "dynr"
    if hidden not in HIDDEN_SIZES:
        raise ValueError(f"hidden must be one of {HIDDEN_SIZES}, got {hidden}")
    return "dynr" if lpt_chunks(T, sm_count, arch, hidden=hidden) is None else "lpt"


def k6_grid(
    T, sm_count, hidden=HIDDEN
):  # static token chunks for the parameter partials -> (G6, R)
    groups = -(-T // 8)
    G6 = min(_tab("K6_CTAS_PER_SM", hidden) * sm_count, groups)
    R = 8 * (-(-groups // G6))
    return G6, R


def row_grid(
    T, sm_count, grid_per_sm
):  # K3, K4, K5: the design's persistent grid (M1 form; a GRID_PER_SM > 0 entry)
    return min(T, grid_per_sm * sm_count)


def k2_grid(T):
    return -(-T // 16)


# The M2 production forms of the row streams launch one trip per CTA (the modules' ``grid_rule`` with
# ``GRID_PER_SM = 0``): ``ceil(T col_split / rows_per_cta)`` CTAs of 256 threads, the grid of ``module.grid_for``.
def row_cta_grid(
    T, rows_per_cta=1, col_split=1
):  # one (sub-)token per thread group and trip
    return -(-(T * col_split) // rows_per_cta)


def k3_grid(
    T, hidden=HIDDEN
):  # fwd_collapse (rows_per_cta=1, col_split=2 at H 4096): 2 T CTAs; ceil(T H / 2048)
    return row_cta_grid(T, *_tab("K3_TILING", hidden))


def k4_grid(T):  # fwd_finish production (rows_per_cta=4, col_split=2): ceil(T / 2) CTAs
    return row_cta_grid(T, 4, 2)


def t_pad(T: int) -> int:
    """``T_pad = 128 ceil(T / 128)``, the row count of the padded records."""
    return TILE_T * (-(-T // TILE_T))


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


@functools.lru_cache(maxsize=4096)
def host_plan(
    rows: int, sm_count: int, arch: str, hidden: int = HIDDEN
) -> dict[str, Any]:
    """Every host-rule output of one ``(T, SM count, architecture, hidden size)``:
    the token padding, the P_K band of the architecture, the
    K-unit count and grid of the mapping pass, the coefficient grid, the chunk
    count, grid and rows per chunk of the column-slab pass, the token-owner
    chunking and the flattened token ``n = 4 hidden``.  ``rows`` must be positive."""

    rows = int(rows)
    sm_count = int(sm_count)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    hidden = int(hidden)
    if hidden not in HIDDEN_SIZES:
        raise ValueError(f"hidden must be one of {HIDDEN_SIZES}, got {hidden}")
    p_k, units1, grid1 = k1_units(rows, sm_count, arch, hidden)
    parts, units7, grid7, rpc = k7_chunks(rows, sm_count, hidden=hidden)
    g6, r6 = k6_grid(rows, sm_count, hidden)
    padded = t_pad(rows)
    return {
        "rows": rows,
        "sm_count": sm_count,
        "arch": arch,
        "hidden": hidden,
        "n": STREAMS * hidden,
        "t_pad": padded,
        "m_tiles": padded // TILE_T,
        "p_k": p_k,
        "log2_pk": int(math.log2(p_k)),
        "k1_units": units1,
        "k1_grid": grid1,
        "k2_grid": k2_grid(rows),
        "k3_grid": k3_grid(rows, hidden),
        "k4_grid": k4_grid(rows),
        "k5_grid": k5_grid(rows, sm_count, arch, hidden),
        "parts": parts,
        "k7_units": units7,
        "k7_grid": grid7,
        "rows_per_chunk": rpc,
        "chunks": g6,
        "rows_per_cta": r6,
    }


def slab_plan(
    rows: int, sm_count: int, arch: str, hidden: int = HIDDEN
) -> dict[str, int]:
    """The chunks of the column slab's ``lpt`` record of hidden size ``hidden`` for ``rows`` tokens:
    :func:`lpt_chunks`'s two-size geometry of the size, else the one-size chunks of :func:`k7_chunks` at the size
    (``lpt_big_parts == lpt_parts``; always these where the size's calibration is ``None``, as on sm_103a and
    sm_107a at 5120 and 7168).  A size :data:`HIDDEN_SIZES` does not list is a ``ValueError`` (as in
    :func:`host_plan`).  Read at call time (not cached: :func:`_lpt_plan` caches the screen) and kept out of
    :func:`host_plan`, which the workspace envelope reads at every 64-token tile.  A geometry that is not a two-size
    partition of the rows is a ``ValueError``."""

    rows, sm_count, hidden = int(rows), int(sm_count), int(hidden)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if hidden not in HIDDEN_SIZES:
        raise ValueError(f"hidden must be one of {HIDDEN_SIZES}, got {hidden}")
    parts, _, _, rows_per_chunk = k7_chunks(rows, sm_count, hidden=hidden)
    geometry = lpt_chunks(rows, sm_count, arch, hidden=hidden) or (
        parts,
        rows_per_chunk,
        parts,
        rows_per_chunk,
    )
    P, rpc, p_big, rpc_small = (int(value) for value in geometry)
    if not (
        1 <= p_big <= P <= 32
        and rpc > 0
        and rpc % SLAB_TILE_T == 0
        and rpc_small > 0
        and rpc_small % SLAB_TILE_T == 0
        and p_big * rpc + (P - p_big) * rpc_small >= rows
    ):
        raise ValueError(
            f"lpt_chunks gave {tuple(geometry)} for {rows} rows: not a two-size geometry "
            "(1 <= p_big <= P <= 32, positive multiples of 64 tokens covering the rows)"
        )
    return {
        "lpt_parts": P,
        "lpt_rows_per_chunk": rpc,
        "lpt_big_parts": p_big,
        "lpt_small_rows_per_chunk": rpc_small,
    }


_DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "uint32": 4}


def workspace_regions(
    kind: str, rows: int, sm_count: int, arch: str, hidden: int = HIDDEN
) -> dict[str, Any]:
    """Byte layout of the caller-owned workspace of ``kind`` (``"fwd"`` or
    ``"bwd"``) for exactly ``rows`` tokens of hidden size ``hidden`` on a device of
    ``sm_count`` SMs and architecture ``arch`` (its P_K rule):
    ``{"regions": {name: {"offset", "bytes", "dtype", "shape"}}, "total_bytes"}``
    with every region at a 256-byte aligned offset.  ``rows == 0`` has no
    regions (nothing is launched)."""

    if kind not in WORKSPACE_REGIONS:
        raise ValueError(
            f"unknown workspace {kind!r}; the workspaces are {sorted(WORKSPACE_REGIONS)}"
        )
    rows = int(rows)
    if rows < 0:
        raise ValueError("rows must be non-negative")
    regions: dict[str, dict[str, Any]] = {}
    if rows == 0:
        return {"regions": regions, "total_bytes": 0}
    plan = host_plan(rows, sm_count, arch, hidden)
    offset = 0
    for region in WORKSPACE_REGIONS[kind]:
        shape = tuple(
            int(plan[dim]) if isinstance(dim, str) else int(dim)
            for dim in region["shape"]
        )
        nbytes = _DTYPE_BYTES[region["dtype"]] * math.prod(shape)
        regions[region["name"]] = {
            "offset": offset,
            "bytes": nbytes,
            "dtype": region["dtype"],
            "shape": shape,
        }
        offset += _round256(nbytes)
    return {"regions": regions, "total_bytes": offset}


def workspace_bytes(
    kind: str, rows: int, sm_count: int, arch: str, hidden: int = HIDDEN
) -> int:
    """Bytes of the ``kind`` workspace for exactly ``rows`` tokens of hidden size
    ``hidden`` (0 for ``rows == 0``)."""

    return int(workspace_regions(kind, rows, sm_count, arch, hidden)["total_bytes"])


@functools.lru_cache(maxsize=4096)
def workspace_bytes_upto(
    kind: str, rows: int, sm_count: int, arch: str, hidden: int = HIDDEN
) -> int:
    """Bytes of a ``kind`` workspace that serves every token count ``1..rows`` of
    hidden size ``hidden``.

    The exact size is not monotone in the token count: the K-unit count of the
    mapping pass and the chunk count of the column-slab pass step down as the
    token count grows.  Within one token tile the unit rule can only step down
    as ``T`` grows (its cost term ``100 * P_K * T`` penalises larger ``P_K``), so
    the forward maximum over a tile is at the tile's first token count; the
    backward regions are functions of the 64-token tile count and of
    ``ceil(T / 8)`` only, non-decreasing within a tile, so its maximum over a
    tile is at the tile's last token count.  The envelope is the maximum over
    the tiles up to ``rows``; a buffer of this size therefore serves any
    ``T <= rows`` and CUDA Graph replays at smaller ``T``.  The argument holds at
    every hidden size: ``hidden`` scales the N-sized regions and the column-slab
    pass's slab count, not the structure of either cost model.
    """

    rows = int(rows)
    if rows <= 0:
        return 0
    if kind == "fwd":
        probes = [(k - 1) * TILE_T + 1 for k in range(1, -(-rows // TILE_T) + 1)]
    else:
        probes = [
            min(k * SLAB_TILE_T, rows) for k in range(1, -(-rows // SLAB_TILE_T) + 1)
        ]
    return max(workspace_bytes(kind, probe, sm_count, arch, hidden) for probe in probes)


def rules_row(T: int, sm_count: int, arch: str, hidden: int = HIDDEN) -> dict[str, Any]:
    """Every host-rule output for one ``(T, sm_count, arch, hidden)`` in the column
    vocabulary of the kernels' test references (their rules table), the program
    columns included (``fn_split_program`` / ``fn_split_grid`` ... ``k5_program``,
    ``fwd_finish_program``; ``k4_grid`` is the residual mixing's grid of
    record): the deterministic package's row without its column-slab and
    reduction program columns (this package's column-slab selection is its own,
    :func:`bwd_slab_atomic_program`; no reduction), with this package's
    ``bwd_workspace_bytes``.  ``T == 0`` is the no-launch row (its T-dependent
    program columns are empty)."""

    row: dict[str, Any] = {
        "T": T,
        "sm_count": sm_count,
        "arch": arch,
        "hidden": hidden,
        "T_pad": t_pad(T),
    }
    programs = dict(
        fn_split_program=fn_split_program(
            arch, 0, hidden
        ),  # the fnp route (fwd_prepare / fwd_recompute)
        fn_split_program_t=fn_split_program(
            arch, 1, hidden
        ),  # the fnTp route (bwd_finish)
        fn_split_grid=fn_split_grid(
            arch, 0, hidden
        ),  # the fnp route's launch grid (one grid per program in both layouts)
        fn_split_grid_t=fn_split_grid(arch, 1, hidden),  # the fnTp route's launch grid
        fwd_coef_program=fwd_coef_program(T, sm_count, arch, hidden) if T else "",
        fwd_map_program=fwd_map_program(T, sm_count, arch, hidden) if T else "",
        bwd_token_program=bwd_token_program(T, sm_count, arch, hidden) if T else "",
        k5_program=k5_program(T, sm_count, arch, hidden)
        if T
        else "",  # the backward prepare's CTA form of record
        fwd_finish_program=fwd_finish_program(arch, hidden),
    )
    if T == 0:
        row.update(
            launch=False,
            P_K=0,
            k1_units=0,
            k1_grid=0,
            P=0,
            k7_units=0,
            k7_grid=0,
            rows_per_chunk=0,
            G6=0,
            R6=0,
        )
        row.update(
            k2_grid=0,
            row_grid4=0,
            k3_grid=0,
            k4_grid=0,
            k5_grid=0,
            fwd_workspace_bytes=0,
            bwd_workspace_bytes=0,
        )
        row.update(programs)
        return row
    plan = host_plan(T, sm_count, arch, hidden)
    row.update(
        launch=True,
        P_K=plan["p_k"],
        k1_units=plan["k1_units"],
        k1_grid=plan["k1_grid"],
        P=plan["parts"],
        k7_units=plan["k7_units"],
        k7_grid=plan["k7_grid"],
        rows_per_chunk=plan["rows_per_chunk"],
        G6=plan["chunks"],
        R6=plan["rows_per_cta"],
        k2_grid=plan["k2_grid"],
        row_grid4=row_grid(T, sm_count, 4),
        k3_grid=k3_grid(T, hidden),
        k4_grid=k4_grid_of_record(T, arch, hidden),
        k5_grid=k5_grid(T, sm_count, arch, hidden),
        fwd_workspace_bytes=workspace_bytes("fwd", T, sm_count, arch, hidden),
        bwd_workspace_bytes=workspace_bytes("bwd", T, sm_count, arch, hidden),
    )
    row.update(programs)
    return row


def rules_table(
    T_values: Sequence[int] = RULES_T_VALUES,
    boards: Sequence[tuple[str, int]] = RULES_BOARDS,
    hiddens: Sequence[int] = HIDDEN_SIZES,
) -> list[dict]:
    """The rows of :func:`rules_row` for every hidden size of ``hiddens`` (in that
    order), every ``T`` (ascending, duplicates dropped) and every
    ``(architecture, SM count)`` board: the table the rule copies are checked
    against."""

    return [
        rules_row(T, sm, arch, h)
        for h in hiddens
        for T in sorted(set(T_values))
        for arch, sm in boards
    ]


# ----------------------------------------------------------------------------
# Routes


def route_key(hidden: int, kind: str) -> str:
    """Name the route of one public call: ``<kind>_h<hidden>``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return f"{kind}_h{int(hidden)}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes with every route of :data:`KINDS` on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        hidden
        for hidden in HIDDEN_SIZES
        if all(route_key(hidden, kind) in routes for kind in KINDS)
    )


def route_applies(*, device_capability: Sequence[int], hidden: int, kind: str) -> bool:
    """Whether the verified export owns one training call."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return route_key(hidden, kind) in ROUTES.get(arch, {})


def route_modules(arch: str, hidden: int, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(hidden, kind)
    names = ROUTES.get(arch, {}).get(key)
    if not names:
        raise ValueError(
            f"unsupported Cake mHC training route {key!r} on {arch!r}; exported "
            f"hidden sizes on this architecture: {supported_hidden_sizes(arch)}"
        )
    return tuple(names)


SELECT_RULE_KINDS = (
    "rows_between",
    "ctas_band",
    "units_fit",
    "chunk_rows_band",
    "program",
    "run_time_iters",
)
# The roles whose compiled programs unroll the Sinkhorn chain for the count of
# record: a call at another count needs the role's run-time-count record.
RUN_TIME_ITERS_ROLES = ("fwd_coef", "bwd_token")
# The host rules a ``program`` selection rule may name (the record's program key must equal the rule's value).
PROGRAM_RULES = (
    "fwd_map_program",
    "fwd_coef_program",
    "bwd_token_program",
    "bwd_slab_atomic_program",
)


def _selects(
    rule: Optional[dict[str, Any]],
    rows: int,
    sm_count: int,
    arch: str,
    sinkhorn_iters: int = SINKHORN_ITERS,
    hidden: int = HIDDEN,
) -> bool:
    if rule is None:
        return True
    kind = rule["kind"]
    if kind not in SELECT_RULE_KINDS:
        raise ValueError(f"unknown Cake mHC training selection rule: {kind!r}")
    if kind == "run_time_iters":
        return int(sinkhorn_iters) != SINKHORN_ITERS
    if kind == "program":
        if rule["rule"] not in PROGRAM_RULES:
            raise ValueError(
                f"unknown Cake mHC training program rule: {rule['rule']!r}"
            )
        return globals()[rule["rule"]](rows, sm_count, arch, hidden) == rule["program"]
    if kind == "chunk_rows_band":
        rows_per_cta = k6_grid(rows, sm_count, hidden)[1]
        return any(
            int(lo) <= rows_per_cta and (hi is None or rows_per_cta <= int(hi))
            for lo, hi in rule["ranges"]
        )
    at_most = rule["at_most"]
    if not (rows > int(rule["above"]) and (at_most is None or rows <= int(at_most))):
        return False
    if kind == "rows_between":
        return True
    if kind == "units_fit":
        return (k1_units(rows, sm_count, arch, hidden)[1] <= sm_count) == bool(
            rule["fit"]
        )
    grid = _ceil_div(rows, int(rule["tile_rows"]))
    lo, hi = rule["lo_ctas_per_sm"], rule["hi_ctas_per_sm"]
    if lo is not None:
        bound = int(lo) * int(sm_count)
        if not (grid >= bound if rule["lo_inclusive"] else grid > bound):
            return False
    if hi is not None:
        bound = int(hi) * int(sm_count)
        if not (grid <= bound if rule["hi_inclusive"] else grid < bound):
            return False
    return True


def selected_modules(
    arch: str,
    hidden: int,
    kind: str,
    *,
    rows: int,
    sm_count: int,
    sinkhorn_iters: int = SINKHORN_ITERS,
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``rows`` tokens
    on a device of ``sm_count`` SMs.

    Each module's recorded ``select_rule`` names the calls it owns
    (``rows_between``: ``above < rows <= at_most``, ``at_most`` ``None`` =
    unbounded; ``units_fit``: the same row range when ``k1_units(rows, sm_count,
    arch).units <= sm_count`` equals its ``fit`` -- the mapping pass's
    one-CTA-per-SM regime; ``ctas_band``: the same row range when ``ceil(rows /
    tile_rows)`` lies in the band ``lo_ctas_per_sm x sm_count .. hi_ctas_per_sm
    x sm_count`` with the recorded end inclusivity (``None`` = unbounded) -- the
    round-3 coefficient-pass regimes keyed on the device's SM count; ``chunk_rows_band``:
    ``k6_grid(rows, sm_count).R`` lies in one of ``ranges`` -- the token-owner
    pass's regimes; ``program``: the named host rule (``fwd_map_program``,
    ``fwd_coef_program``, ``bwd_token_program`` or ``bwd_slab_atomic_program``)
    at ``(rows, sm_count, arch, hidden)`` equals the record's program key --
    the mapping-pass forms of sm_103a and
    sm_100a outside the fit regime (the ``nsx`` record ``__tk`` also where the
    rule names the ticket form ``tk_nsx_ct``, which is not exported and is
    bitwise ``nsx``; the ``ns96_g2_ol2l`` record ``__tk`` likewise for
    ``tk_ns96_g2_ol2l_ct``), the coefficient-pass forms of record and the
    token-owner forms on sm_107a and the column slab's two forms;
    ``run_time_iters``: every call whose
    ``sinkhorn_iters`` is not :data:`SINKHORN_ITERS` -- the run-time-count forms
    of the coefficient pass and the token-owner pass, which then stand in for
    every other record of their ``role``; ``None`` = every call); exactly the
    modules whose rule admits the call are launched, in route order.  Every
    rule reads the record's own hidden size.  A route without a run-time-count
    record for a role of :data:`RUN_TIME_ITERS_ROLES` runs the count of record
    only: another count is a ``NotImplementedError``.  The rules of record are
    :func:`fn_split_program`, :func:`fwd_map_program`, :func:`fwd_coef_program`,
    :func:`bwd_token_program` and :func:`bwd_slab_atomic_program` (the
    deterministic package's selection on every route but the column slab's,
    this package's column-slab rule at every hidden size: on sm_103a at 5120 /
    7168, whose route holds the ``dynr__h<H>`` record alone, the rule names
    ``dynr`` at every token count; the fn split of ``bwd_finish`` is one record
    per architecture and hidden size).
    """

    rows, sinkhorn_iters = int(rows), int(sinkhorn_iters)
    route = route_modules(arch, hidden, kind)
    # At a count other than the count of record, the run-time-count record of a role replaces the role's
    # compiled records (whose chain is unrolled for SINKHORN_ITERS).
    run_time_roles = (
        {
            MODULES[name]["role"]
            for name in route
            if (MODULES[name]["select_rule"] or {}).get("kind") == "run_time_iters"
        }
        if sinkhorn_iters != SINKHORN_ITERS
        else set()
    )
    if sinkhorn_iters != SINKHORN_ITERS:
        compiled_only = sorted(
            {
                MODULES[name]["role"]
                for name in route
                if MODULES[name]["role"] in RUN_TIME_ITERS_ROLES
            }
            - run_time_roles
        )
        if compiled_only:
            raise NotImplementedError(
                f"the {route_key(hidden, kind)!r} route on {arch!r} runs "
                f"sinkhorn_iters={SINKHORN_ITERS} only (no run-time-count program of "
                f"{compiled_only}); got sinkhorn_iters={sinkhorn_iters}"
            )
    names = tuple(
        name
        for name in route
        if (
            (MODULES[name]["select_rule"] or {}).get("kind") == "run_time_iters"
            or MODULES[name]["role"] not in run_time_roles
        )
        and _selects(
            MODULES[name]["select_rule"],
            rows,
            int(sm_count),
            arch,
            sinkhorn_iters,
            MODULES[name]["hidden"],
        )
    )
    if not names:
        raise ValueError(
            f"no exported module of the {route_key(hidden, kind)!r} route on {arch!r} admits {rows} rows"
        )
    return names


# ----------------------------------------------------------------------------
# Programmatic dependent launch per call
#
# The binding of a ``use_pdl == "runtime"`` record takes ``enable_pdl`` (0 / 1) as
# its last host argument before the grid.  ``PDL_OFF_BANDS[arch][edge][path]``
# lists the token-count bands ``(above, at_most]`` where the attribute is off on
# one edge (the dependent kernel after its predecessor) of one call path
# (``"forward"``: the forward prepare; ``"checkpoint"``: the backward after a
# replayed forward prefix; ``"backward"``: the backward from saved statistics);
# elsewhere, and on every edge without a band, the attribute is on.

PDL_PATHS = ("forward", "backward", "checkpoint")
# Library stem -> the edge its records sit on (predecessor -> dependent).  The
# bands are the deterministic package's on the edges this package has (the
# atomic column slab (both forms), launched behind the fn split that the
# token-owner pass releases at once, takes the deterministic column slab's
# K6->K7 bands; there is no K7->K8 edge).
PDL_EDGES = {
    "cake_mhc_train_bwd_prepare": "K1->K5",
    "cake_mhc_train_bwd_slab_atomic": "K6->K7",
    "cake_mhc_train_fwd_collapse": "K1->K3",
}
PDL_OFF_BANDS: dict[str, dict[str, dict[str, tuple[tuple[int, int], ...]]]] = {
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
    },
    "sm_107a": {},
}


def pdl_edge(library: str) -> str | None:
    """The edge of one library's records (``None`` for a kernel outside :data:`PDL_EDGES`)."""

    return PDL_EDGES.get(library.split("__", 1)[0])


def enable_pdl(name: str, rows: int, arch: str, path: str) -> int:
    """The ``enable_pdl`` argument of one launch of record ``name`` for ``rows`` tokens
    on ``arch`` along ``path``: 0 inside a band of :data:`PDL_OFF_BANDS` of the
    record's edge, 1 elsewhere (a record whose ``use_pdl`` is not ``"runtime"``
    takes no such argument and reads 1)."""

    if path not in PDL_PATHS:
        raise ValueError(f"unknown Cake mHC training call path: {path!r}")
    record = MODULES[name]
    if record["launch"]["use_pdl"] != "runtime":
        return 1
    edge = pdl_edge(record["library"])
    if edge is None:
        raise ValueError(
            f"record {name!r} takes the per-call attribute but its library sits on no edge of PDL_EDGES"
        )
    rows = int(rows)
    for above, at_most in PDL_OFF_BANDS.get(arch, {}).get(edge, {}).get(path, ()):
        if int(above) < rows and (at_most is None or rows <= int(at_most)):
            return 0
    return 1


def workspace_kind(kind: str) -> str | None:
    """The workspace a route holds (``"fwd"`` / ``"bwd"``), or ``None``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return WORKSPACE_OF_KIND.get(kind)


# ----------------------------------------------------------------------------
# Launch


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``fixed`` modules launch exactly ``ctas`` CTAs (the fn splits).  ``tiles``
    modules launch ``ceil(rows / rows_per_tile)`` CTAs
    (the coefficient pass).  ``rows_per_cta`` modules launch
    ``ceil(rows * col_split / rows_per_cta)`` CTAs, capped at ``grid_limit``
    when positive, else at ``grid_per_sm * SM count`` when positive (the row
    streams, one (sub-)token per thread group and trip: :func:`k3_grid` =
    ``2 rows`` for the collapse, :func:`k4_grid` = ``ceil(rows / 2)`` for the
    residual mixing, :func:`k5_grid` for the backward prepare: ``ceil(rows / 2)``,
    capped at ``grid_per_sm x SM count`` inside a band of ``K5_GRID_BANDS`` --
    2 x SMs for ``rows <= 4096`` on sm_103a and for
    ``rows <= 1024`` on sm_100a, the band record's ``grid_per_sm``; no cap on
    the other records).  ``units_bounded`` modules launch
    ``min(ctas_per_sm * SM count, ceil(rows / tile_rows) * P_K)`` persistent
    CTAs over the (token tile, K unit) work units by static grid-stride (the
    mapping pass; ``P_K`` from :func:`k1_units`; the ticket region is bound for
    the opt-in ticket schedule and left unread by the static form).  ``chunks`` modules
    launch ``min(ctas_per_sm * SM count, ceil(rows / group_rows))`` CTAs, each
    owning a static run of token groups (the token-owner pass).  ``slab_chunks``
    modules launch ``min(SM count, slabs * P)`` CTAs over (column slab, token
    chunk) units (the column-slab pass; ``P`` from :func:`k7_chunks`, or
    :func:`slab_plan`'s ``lpt_parts`` for the ``lpt_chunks`` rule; both forms
    draw the units from the backward workspace's self-resetting ticket word).
    Every rule reads the record's hidden size (``record["hidden"]``): ``P_K``,
    ``P`` and the fixed grids and slab counts of the records follow it.
    ``rows`` must be positive; the public API returns empty outputs for zero
    rows without launching.
    """

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "fixed":
        grid_x = int(rule["ctas"])
    elif kind == "tiles":
        grid_x = _ceil_div(rows, int(rule["rows_per_tile"]))
    elif kind == "rows_per_cta":
        grid_x = _ceil_div(
            rows * int(rule.get("col_split", 1)), int(rule["rows_per_cta"])
        )
        limit = int(rule.get("grid_limit", 0))
        per_sm = int(rule.get("grid_per_sm", 0))
        if limit <= 0 and per_sm > 0:
            limit = per_sm * sm_count(device_index)
        if limit > 0:
            grid_x = min(grid_x, limit)
    elif kind == "units_bounded":
        plan = host_plan(
            rows, sm_count(device_index), device_arch(device_index), record["hidden"]
        )
        units = _ceil_div(rows, int(rule["tile_rows"])) * plan["p_k"]
        grid_x = min(int(rule["ctas_per_sm"]) * sm_count(device_index), units)
    elif kind == "chunks":
        grid_x = min(
            int(rule["ctas_per_sm"]) * sm_count(device_index),
            _ceil_div(rows, int(rule["group_rows"])),
        )
    elif kind == "slab_chunks":
        sms, arch = sm_count(device_index), device_arch(device_index)
        if rule["chunks_rule"] == "lpt_chunks":
            parts = slab_plan(rows, sms, arch, hidden=record["hidden"])["lpt_parts"]
        else:
            parts = host_plan(rows, sms, arch, record["hidden"])["parts"]
        grid_x = min(sms, int(rule["slabs"]) * parts)
    else:
        raise ValueError(f"unknown Cake mHC training grid rule: {kind!r}")
    return (max(grid_x, 1), 1, 1)


def workspace_layout(
    kind: str, rows: int, *, device_index: int, hidden: int = HIDDEN
) -> dict[str, Any]:
    """:func:`workspace_regions` of ``kind`` for ``rows`` tokens of hidden size
    ``hidden`` on ``device_index``."""

    return workspace_regions(
        kind, rows, sm_count(device_index), device_arch(device_index), hidden
    )


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake mHC training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake mHC training module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    from . import cake_build

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    from . import cake_build

    return cake_build.jit_specs(arch)


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``
    to tensors and Python scalars; the module record maps its kernel parameter
    names onto that vocabulary, and a scalar the record fixes itself
    (:data:`RECORD_SCALARS`) overrides the caller's value.  The launch grid
    follows the module's recorded grid rule for ``rows`` tokens and the kernel
    runs on the current torch stream.  A record whose binding takes the
    per-call PDL argument reads ``values["enable_pdl"]`` (default 1, the
    attribute on; the public API decides it per call with :func:`enable_pdl`).
    """

    import tvm_ffi

    record = MODULES[name]
    logical = record["logical"]
    fixed = RECORD_SCALARS.get(name, {})
    grid = launch_grid(record, rows, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        if kind == "launch":
            args.append(int(values.get("enable_pdl", 1)))
            continue
        logical_name = logical[key]
        if logical_name in fixed:
            args.append(fixed[logical_name])
            continue
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "DZP_COLS",
    "GSTATS_COLS",
    "GSTATS_LAYOUT",
    "HC_EPS",
    "HIDDEN",
    "HIDDEN_SIZES",
    "HIDDEN_TOTAL",
    "H_TABLES",
    "H_TABLE_NAMES",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAPPING_ROWS",
    "MODULES",
    "N_PAD",
    "PDL_EDGES",
    "PDL_OFF_BANDS",
    "PDL_PATHS",
    "PARAM_COLS",
    "PARAM_LAYOUT",
    "PIECES",
    "RECORD_SCALARS",
    "RMS_EPS",
    "ROUTES",
    "PROGRAM_RULES",
    "RULES_BOARDS",
    "RULES_SM_COUNTS",
    "RULES_T_VALUES",
    "SELECT_RULE_KINDS",
    "SINKHORN_ITERS",
    "SLAB_D",
    "SLAB_TILE_T",
    "STATS_COLS",
    "STATS_LAYOUT",
    "STREAMS",
    "TICKET_WORDS",
    "TILE_T",
    "TOKP_COLS",
    "TOKP_LAYOUT",
    "WORKSPACE_OF_KIND",
    "WORKSPACE_REGIONS",
    "BWD_SLAB_LPT",
    "BWD_SLAB_LPT_HIDDEN",
    "BWD_TOKEN_BANDS",
    "BWD_TOKEN_PROGRAM",
    "BWD_TOKEN_REGIMES",
    "FN_SPLIT_GRID",
    "FN_SPLIT_PROGRAM",
    "FWD_COEF_REGIMES",
    "FWD_FINISH_PROGRAM",
    "FWD_FINISH_ROWS",
    "FWD_MAP_BANDS",
    "FWD_MAP_CLAIM",
    "FWD_MAP_FIT",
    "FWD_MAP_PRODUCTION",
    "K5_FORM_BANDS",
    "K5_FORM_ROWS",
    "K5_GRID_BANDS",
    "PK_BANDS",
    "FWD_FINISH_COL_SPLIT",
    "K3_TILING",
    "K6_CTAS_PER_SM",
    "RUN_TIME_ITERS_ROLES",
    "arch_for_capability",
    "bwd_slab_atomic_program",
    "bwd_token_program",
    "device_arch",
    "enable_pdl",
    "fn_split_grid",
    "fn_split_program",
    "fwd_coef_program",
    "fwd_finish_program",
    "fwd_map_program",
    "host_plan",
    "k1_units",
    "k4_grid_of_record",
    "k2_grid",
    "k3_grid",
    "k4_grid",
    "k5_grid",
    "k5_program",
    "k6_grid",
    "k7_chunks",
    "launch_grid",
    "library_name",
    "load",
    "lpt_chunks",
    "pdl_edge",
    "route_applies",
    "route_key",
    "route_modules",
    "row_cta_grid",
    "row_grid",
    "rules_row",
    "rules_table",
    "run_module",
    "selected_modules",
    "slab_plan",
    "sm_count",
    "specs_for_arch",
    "supported_hidden_sizes",
    "t_pad",
    "workspace_bytes",
    "workspace_bytes_upto",
    "workspace_kind",
    "workspace_layout",
    "workspace_regions",
]
