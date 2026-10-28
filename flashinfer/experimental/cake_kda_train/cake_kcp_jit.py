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

Registry and loader glue of the Cake packed KDA chunked training kernels on
the context-parallel (KCP) route (SM100 / SM103 / SM107): the state walks,
the transfer walk and the affine merge that :mod:`.cake_kcp` composes with the
chunk stages of :mod:`.cake_jit`, whose records this module never edits.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``, beside the chunk
stages' entries), the architectures it is delivered for, the KCP ``program``
it serves and its ``variant``, the ``select_rule`` that picks one variant of a
program from the launch's host-known scalars, the ``grid_rule`` that shapes
its launch, the kernel-parameter -> logical-argument map and the argument plan
of its ``run`` entry.  What to compile -- sources, digests, nvcc flags -- is
recorded once in ``build.json`` next to this file; :mod:`.cake_build` serves
the library from a prebuilt ``aot/`` file when a builder compiled it and
otherwise through FlashInfer's JIT from the same sources and flags.

The programs are the eight launches of the KCP composition
(:data:`PROGRAMS`, the names :mod:`.cake_kcp` passes to :func:`run_program`):

* ``kcp_fwdh_offset`` / ``kcp_fwdh_tape`` -- the forward state recurrence
  walked over the segments a host table lists, from a compact indexed initial
  state; the offset program writes each final state to its indexed tile, the
  tape program writes the ``h`` / ``v_new`` tapes.  Each program ships in two
  records (``select_rule`` ``walk_form``): the full-width form (one CTA per
  segment and head over the 128 value columns; ``grid_rule``
  ``kcp_segments_heads``: ``n_walk * num_heads`` CTAs) and its ``_vh64`` form
  (two CTAs per segment and head over 64 value columns each,
  ``ctas_per_head`` 2), bitwise equal; :func:`walk_form` picks the form of a
  launch from its walk list;
* ``kcp_dhu_offset`` / ``kcp_dhu_tape`` -- the adjoint recurrence likewise
  (reverse walk; the tape program writes the ``dh`` / ``dv2`` tapes), in the
  same two forms;
* ``kcp_transfer_walk`` -- the FP32-class transfer ``M_f`` (``^T``) of a
  segment (BF16x3 on the tensor cores: two CTAs per segment and head, each
  over one 64-column half of the state) or, with ``init_mode = 1``, its
  FP32-class offset;
* ``kcp_affine_merge`` -- the exact FP32 affine chains over tile pools
  (``grid_rule`` ``kcp_chains``: ``n_chains * (ncols / cols_per_cta)`` CTAs);
  two variants by the chain's column count (``select_rule`` ``columns``:
  ``cols_per_cta`` 8 for 128-column state chains, 16 for 256-column map
  chains, the faster build of each on GB300);
* ``kcp_fwd_o`` -- the chunk output stage of the forward at the rank's
  shape: ``out`` from the ``h`` / ``v_new`` tapes the forward walk writes and
  the intra stage's ``qg`` / ``Aqk`` rows, persistent CTAs over the (chunk,
  head) tiles (``grid_rule`` ``kcp_sm_bounded``: ``min(num_chunks_pad *
  num_heads, ctas_per_sm * sm_count)`` CTAs, at least one);
* ``kcp_dav`` -- the output backward at the rank's shape: ``dAqk`` and the
  first value gradient ``dv1`` (read by the adjoint walk) from ``do_int``, the
  ``v_new`` tape and the ``Aqk`` rows, one CTA per chunk pair and head
  (``grid_rule`` ``kcp_pairs``: ``(num_chunks_pad / 2, num_heads, 1)``).

Compact state pools (the walks' ``h0`` / ``dht`` inputs and ``out`` outputs)
are flat FP32 tensors of ``128 x 128`` tiles addressed by an int32 slot table
``index[num_seqs]`` (``-1``: the zero state on input, nothing written on
output) and two integers ``hg`` / ``n``: head ``h`` of slot ``s`` is tile
``((h // hg) * n + s) * hg + h % hg``, placed ``tile * tile_stride * 16384``
floats after the pool's base pointer (``tile_stride`` 1: a dense plane of
tiles; 2: the exchange's interleaved pair layout, the half selected by the
base pointer).  :mod:`.cake_kcp` spells a pool as an ``Indexed`` (pool, slot
table, tile table); :func:`run_program` recovers the two-integer form through
``Indexed.addressing`` and lowers every launch of the composition -- the walk
lists, slot tables and chain tables to cached device copies, the pools to
base-pointer views, the operand and tape tensors from the call's local stage
values -- onto :func:`run_module`.

Test-only record set (``build.test.json``, loaded only after
:func:`cake_kcp_plan.enable_test_knobs`): the TF32 hi/lo transfer walk
(``transfer_walk`` knob ``tf32_hilo``: the previous shipped program, one CTA
per segment and head) with its static ``precision_mode`` forms 1 (the
explicit three-MMA TF32x3 form of the bitwise unit test) and 2 (plain TF32, a
cross-check that is not a gate), and the full-width BF16x3 walk
(``transfer_walk`` knob ``bf16x3``: one CTA per segment and head over all 128
value columns) for the A/B record of the shipped half-width program; their
records are ``TEST_MODULES`` and their libraries never enter ``build.json``,
so the shipped route carries the BF16x3 half-width program only.  The public
API cannot select them: :mod:`.cake_kcp` takes ``precision_mode`` and
``transfer_walk`` from the validated ``_kcp_test_knobs`` option alone.  The
``walk_form`` knob (:data:`WALK_FORMS`) forces the form of every K1 / K2 walk
launch (``full`` / ``split``) for the A/B record; both forms are shipped
records, so it names no test-only library.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import tvm_ffi

from . import cake_build, cake_jit, cake_kcp_plan

# One verified build per architecture: B200 (``sm_100a``), B300 / GB300
# (``sm_103a``) and R200 (``sm_107a``); the chunk stages' architectures.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")

# The launches of the KCP composition, by the names :mod:`.cake_kcp` uses
# (``cake_kcp.KCP_PROGRAMS``).  The first five are walks over a segment list;
# the sixth composes chains; the last two are the chunk stages of the forward
# output and the output backward at the rank's shape.
PROGRAMS = (
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_dhu_offset",
    "kcp_dhu_tape",
    "kcp_transfer_walk",
    "kcp_affine_merge",
    "kcp_fwd_o",
    "kcp_dav",
)
WALK_PROGRAMS = PROGRAMS[:5]
# Elements of one FP32 state tile [128, 128].
TILE = 128 * 128
# The static ``precision_mode`` knob of the TF32 hi/lo transfer walk: 0 the
# program the ``transfer_walk`` knob selects, 1 / 2 the test-only TF32
# programs of ``build.test.json``.
PRECISION_MODES = (0, 1, 2)
# The ``transfer_walk`` knob: ``bf16x3_hw64`` the shipped program (BF16x3, two
# CTAs per segment and head), ``tf32_hilo`` / ``bf16x3`` the test-only programs
# of ``build.test.json`` (the TF32 hi/lo walk and the full-width BF16x3 form).
TRANSFER_WALKS = ("bf16x3_hw64", "tf32_hilo", "bf16x3")
# The ``walk_form`` test knob of the K1 / K2 state walks: ``auto`` the shipped
# rule (:func:`walk_form`), ``full`` / ``split`` force the full-width or the
# two-CTA record of every walk launch (the A/B record).
WALK_FORMS = ("auto", "full", "split")
# The shipped form rule per walk program: ``fit`` = the two-CTA record when the
# launch list-schedules within its longest chain (:func:`walk_form`), else the
# full-width one (the offset programs: two shipped records); ``split`` = the
# two-CTA record on every launch (the tape programs, whose per-step tape stores
# halve with the CTA's columns: their full-width records are test-only, under
# ``walk_form="full"``).
WALK_FORM_RULES = {
    "kcp_fwdh_offset": "fit",
    "kcp_fwdh_tape": "split",
    "kcp_dhu_offset": "fit",
    "kcp_dhu_tape": "split",
}
# The fit rule's step ratio: a two-CTA step beside a co-resident CTA, relative
# to the full-width step of a CTA alone on its SM (measured 0.74-0.84 on the
# one-wave lists; 0.9 is the conservative constant of the rule).
WALK_FORM_FIT_STEP_RATIO = 0.9
# Launches per form since import (evidence of the rule's choices).
WALK_FORM_STATS = {"full": 0, "split": 0}
TEST_MANIFEST = "build.test.json"

# Logical argument names :func:`run_program` binds; every record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).  The
# operand and tape names are the chunk stages' (``cake_jit.LOGICAL_TENSORS``):
# BF16 ``w`` / ``kg`` / ``u`` / ``qg`` / ``do_int`` / ``dv1`` ``[R, H, 128]``
# and FP32 ``gk`` from the prep stages, the BF16 tapes ``h`` / ``dh``
# ``[NC, H, 128, 128]`` and ``v_new`` / ``dv2`` ``[R, H, 128]``.  The KCP
# names: ``seg_chunk_start`` the int32 ``[num_seqs + 1]`` chunk-aligned
# segment table of the rank's fragments, ``walk_seg`` the int32 segment list
# of the launch, the FP32 pools ``h0`` / ``dht`` (initial states and final
# adjoints) and ``out`` (final states, adjoints or transfers) with their int32
# slot tables ``h0_index`` / ``dht_index`` / ``out_index``; the merge's FP32
# tile pools ``m`` / ``c`` / ``x0`` / ``y`` and its chain tables
# ``chain_start`` (int32 ``[n_chains + 1]``), ``elem_m`` / ``elem_c`` /
# ``elem_y`` (int64 element offsets per element), ``x0_index`` / ``y_final``
# (int64 ``[n_chains]``).  The chunk stages' names: ``h2d`` the ``[NC * H *
# 128, 128]`` view of the ``h`` tape, ``aqk_int`` the BF16 internal ``Aqk``
# rows ``[R, H, 64]`` and ``dAqk`` their FP32 gradient, ``out`` (bound by
# ``kcp_fwd_o``) the call's BF16 output ``[T, H, 128]``, and the int32 chunk
# tables ``chunk_bos`` / ``chunk_len`` (``[NC_pad]``) / ``seq_chunk_start``
# (``[num_seqs + 1]``, the chunk table of the rank's sequences, not the KCP
# segment table ``seg_chunk_start``).
LOGICAL_TENSORS = (
    "w",
    "kg",
    "u",
    "gk",
    "qg",
    "do_int",
    "dv1",
    "h",
    "v_new",
    "dh",
    "dv2",
    "h2d",
    "aqk_int",
    "dAqk",
    "chunk_bos",
    "chunk_len",
    "seq_chunk_start",
    "seg_chunk_start",
    "walk_seg",
    "h0",
    "h0_index",
    "dht",
    "dht_index",
    "out",
    "out_index",
    "m",
    "c",
    "x0",
    "y",
    "chain_start",
    "elem_m",
    "elem_c",
    "elem_y",
    "x0_index",
    "y_final",
)
LOGICAL_SCALARS = (
    "num_heads",
    "num_seqs",
    "scale",
    "h0_hg",
    "h0_n",
    "dht_hg",
    "dht_n",
    "out_hg",
    "out_n",
    "tile_stride",
    "init_mode",
    "use_u",
    "write_transposed",
    "ncols",
    "transpose_m",
    "write_states",
    "cols_per_cta",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``, ``launch``, ``arg_plan``,
# ``logical``); the host plan (``program``, ``variant``, ``select_rule``, ``grid_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal: FlashInfer's manifest checker and the exporter read this
# file with ``ast.literal_eval``.  Record names are the programs, suffixed with the variant where a program has
# several records.
MODULES: dict[str, dict[str, Any]] = {
    "kcp_fwdh_offset": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_7658fcc297c6fd906d53",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_7658fcc297c6fd906d53",
        "logical": {
            "gk": "gk",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_offset",
        "select_rule": {"form": "full", "kind": "walk_form"},
        "variant": None,
    },
    "kcp_fwdh_offset_vh64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_5a20cd12bb62bb6f3ac5",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "use_pdl": False,
        },
        "library": "cake_kda_train_5a20cd12bb62bb6f3ac5",
        "logical": {
            "gk": "gk",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_offset",
        "select_rule": {"form": "split", "kind": "walk_form"},
        "variant": "vh64",
    },
    "kcp_fwdh_tape": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_6fac4410ba86c48541c4",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_6fac4410ba86c48541c4",
        "logical": {
            "gk": "gk",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_tape",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "kcp_dhu_offset": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "scale"],
            ["buffer", "walk_seg"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_1c66573e87150245374b",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_1c66573e87150245374b",
        "logical": {
            "dh_tape": "dh",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_offset",
        "select_rule": {"form": "full", "kind": "walk_form"},
        "variant": None,
    },
    "kcp_dhu_offset_vh64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "scale"],
            ["buffer", "walk_seg"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_afe02fb600500ba3556f",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_kda_train_afe02fb600500ba3556f",
        "logical": {
            "dh_tape": "dh",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_offset",
        "select_rule": {"form": "split", "kind": "walk_form"},
        "variant": "vh64",
    },
    "kcp_dhu_tape": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "scale"],
            ["buffer", "walk_seg"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_32b1aa657b082f2f4cb7",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_32b1aa657b082f2f4cb7",
        "logical": {
            "dh_tape": "dh",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_tape",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "kcp_transfer_walk": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # Two CTAs per (segment, head): each walks one 64-column half of the state.
        "grid_rule": {"ctas_per_head": 2, "kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_9a3f306bacf8c8a1547d",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_kda_train_9a3f306bacf8c8a1547d",
        "logical": {
            "gk": "gk",
            "init_mode": "init_mode",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "kcp_affine_merge_c8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m"],
            ["buffer", "c"],
            ["buffer", "x0"],
            ["buffer", "y"],
            ["buffer", "chain_start"],
            ["buffer", "elem_m"],
            ["buffer", "elem_c"],
            ["buffer", "elem_y"],
            ["buffer", "x0_index"],
            ["buffer", "y_final"],
            ["parameter", "ncols"],
            ["parameter", "transpose_m"],
            ["parameter", "write_states"],
            ["parameter", "cols_per_cta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # One CTA per chain and block of ``cols_per_cta`` columns; the 8-column
        # build is the fastest of the three on 128-column state chains (GB300).
        "grid_rule": {"cols_per_cta": 8, "kind": "kcp_chains"},
        "kernel_symbol": "kernel_cake_kda_train_c6958a0a4ed5cb55a9a1",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 37120,
            "use_pdl": False,
        },
        "library": "cake_kda_train_c6958a0a4ed5cb55a9a1",
        "logical": {
            "c": "c",
            "chain_start": "chain_start",
            "cols_per_cta": "cols_per_cta",
            "elem_c": "elem_c",
            "elem_m": "elem_m",
            "elem_y": "elem_y",
            "m": "m",
            "ncols": "ncols",
            "transpose_m": "transpose_m",
            "write_states": "write_states",
            "x0": "x0",
            "x0_index": "x0_index",
            "y": "y",
            "y_final": "y_final",
        },
        "program": "kcp_affine_merge",
        "select_rule": {"kind": "columns", "ncols": 128},
        "variant": "c8",
    },
    "kcp_affine_merge_c16": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "m"],
            ["buffer", "c"],
            ["buffer", "x0"],
            ["buffer", "y"],
            ["buffer", "chain_start"],
            ["buffer", "elem_m"],
            ["buffer", "elem_c"],
            ["buffer", "elem_y"],
            ["buffer", "x0_index"],
            ["buffer", "y_final"],
            ["parameter", "ncols"],
            ["parameter", "transpose_m"],
            ["parameter", "write_states"],
            ["parameter", "cols_per_cta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # The 16-column build is the fastest of the three on 256-column map
        # chains (``X = [B | M]``; GB300).
        "grid_rule": {"cols_per_cta": 16, "kind": "kcp_chains"},
        "kernel_symbol": "kernel_cake_kda_train_89f5ff59f03eb170e3bf",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 41216,
            "use_pdl": False,
        },
        "library": "cake_kda_train_89f5ff59f03eb170e3bf",
        "logical": {
            "c": "c",
            "chain_start": "chain_start",
            "cols_per_cta": "cols_per_cta",
            "elem_c": "elem_c",
            "elem_m": "elem_m",
            "elem_y": "elem_y",
            "m": "m",
            "ncols": "ncols",
            "transpose_m": "transpose_m",
            "write_states": "write_states",
            "x0": "x0",
            "x0_index": "x0_index",
            "y": "y",
            "y_final": "y_final",
        },
        "program": "kcp_affine_merge",
        "select_rule": {"kind": "columns", "ncols": 256},
        "variant": "c16",
    },
    "kcp_fwd_o": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "qg_tma"],
            ["buffer", "h_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "vn_tma"],
            ["buffer", "o_out"],
            ["buffer", "chunk_bos"],
            ["buffer", "chunk_len"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"ctas_per_sm": 1, "kind": "kcp_sm_bounded"},
        "kernel_symbol": "kernel_cake_kda_train_1214bfc28e50985bfa0c",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "use_pdl": False,
        },
        "library": "cake_kda_train_1214bfc28e50985bfa0c",
        "logical": {
            "aqk_tma": "aqk_int",
            "chunk_bos": "chunk_bos",
            "chunk_len": "chunk_len",
            "h_tma": "h2d",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "o_out": "out",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seq_chunk_start",
            "vn_tma": "v_new",
        },
        "program": "kcp_fwd_o",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
    "kcp_dav": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "do_tma"],
            ["buffer", "vnew_tma"],
            ["buffer", "aqk_tma"],
            ["buffer", "dAqk"],
            ["buffer", "dv1"],
            ["parameter", "num_heads"],
            ["parameter", "scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_pairs"},
        "kernel_symbol": "kernel_cake_kda_train_f227dd21f1f3cb464b29",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_kda_train_f227dd21f1f3cb464b29",
        "logical": {
            "aqk_tma": "aqk_int",
            "dAqk": "dAqk",
            "do_tma": "do_int",
            "dv1": "dv1",
            "num_heads": "num_heads",
            "scale": "scale",
            "vnew_tma": "v_new",
        },
        "program": "kcp_dav",
        "select_rule": {"kind": "always"},
        "variant": None,
    },
}

# The test-only record set: the TF32 hi/lo transfer walk
# ``kda_chunk_kcp_transfer_walk_v1`` (``_tf32_hilo``: the previous shipped
# program, one CTA per segment and head) with its static knob
# ``precision_mode`` 1 (``_three_mma``: the explicit three-MMA TF32x3 form,
# bitwise equal to the hi/lo program) and 2 (``_plain_tf32``: the lo products
# skipped; an optional cross-check, never a gate), and the BF16x3 walk
# ``kda_chunk_kcp_transfer_walk_bf16x3_v1`` at ``vh`` 128 (``_bf16x3``: one
# CTA per segment and head; the shipped ``kcp_transfer_walk`` is its ``vh`` 64
# form, ``grid_rule.ctas_per_head`` 2).  Same kernel signature, argument plan
# and logical map as ``kcp_transfer_walk``; the libraries live in
# ``build.test.json`` only (:func:`load_test`), never in ``build.json``, and
# :func:`run_program` reaches a record only through the call's test knobs
# (``test_knobs``: the knob values that select it).
TEST_MODULES: dict[str, dict[str, Any]] = {
    "kcp_transfer_walk_three_mma": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_b1477ed31fad32a5c2c9",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198656,
            "use_pdl": False,
        },
        "library": "cake_kda_train_b1477ed31fad32a5c2c9",
        "logical": {
            "gk": "gk",
            "init_mode": "init_mode",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "test_knobs": {"precision_mode": 1},
        "variant": "three_mma",
    },
    "kcp_transfer_walk_plain_tf32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_64ba2583f28f7d4c0102",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198656,
            "use_pdl": False,
        },
        "library": "cake_kda_train_64ba2583f28f7d4c0102",
        "logical": {
            "gk": "gk",
            "init_mode": "init_mode",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "test_knobs": {"precision_mode": 2},
        "variant": "plain_tf32",
    },
    "kcp_transfer_walk_bf16x3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_b1a95a8ee8110e47558e",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": False,
        },
        "library": "cake_kda_train_b1a95a8ee8110e47558e",
        "logical": {
            "gk": "gk",
            "init_mode": "init_mode",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "test_knobs": {"transfer_walk": "bf16x3"},
        "variant": "bf16x3",
    },
    "kcp_transfer_walk_tf32_hilo": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["parameter", "init_mode"],
            ["parameter", "use_u"],
            ["parameter", "write_transposed"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_1ac972f0cb7cc7b13cf7",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198656,
            "use_pdl": False,
        },
        "library": "cake_kda_train_1ac972f0cb7cc7b13cf7",
        "logical": {
            "gk": "gk",
            "init_mode": "init_mode",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "use_u": "use_u",
            "w_tma": "w",
            "walk_seg": "walk_seg",
            "write_transposed": "write_transposed",
        },
        "program": "kcp_transfer_walk",
        "select_rule": {"kind": "always"},
        "test_knobs": {"transfer_walk": "tf32_hilo"},
        "variant": "tf32_hilo",
    },
    "kcp_fwdh_tape_full": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "w_tma"],
            ["buffer", "kg_tma"],
            ["buffer", "u"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["buffer", "walk_seg"],
            ["buffer", "h0"],
            ["buffer", "h0_index"],
            ["parameter", "h0_hg"],
            ["parameter", "h0_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "h_tape"],
            ["buffer", "v_new"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_9cfe05bce78410f1304b",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_9cfe05bce78410f1304b",
        "logical": {
            "gk": "gk",
            "h0": "h0",
            "h0_hg": "h0_hg",
            "h0_index": "h0_index",
            "h0_n": "h0_n",
            "h_tape": "h",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "u": "u",
            "v_new": "v_new",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_fwdh_tape",
        "select_rule": {"kind": "always"},
        "test_knobs": {"walk_form": "full"},
        "variant": "full",
    },
    "kcp_dhu_tape_full": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "kg_tma"],
            ["buffer", "qg_tma"],
            ["buffer", "w_tma"],
            ["buffer", "do_tma"],
            ["buffer", "dv1"],
            ["buffer", "gk"],
            ["parameter", "num_heads"],
            ["buffer", "seq_chunk_start"],
            ["parameter", "num_seqs"],
            ["parameter", "scale"],
            ["buffer", "walk_seg"],
            ["buffer", "dht"],
            ["buffer", "dht_index"],
            ["parameter", "dht_hg"],
            ["parameter", "dht_n"],
            ["buffer", "out"],
            ["buffer", "out_index"],
            ["parameter", "out_hg"],
            ["parameter", "out_n"],
            ["parameter", "tile_stride"],
            ["buffer", "dh_tape"],
            ["buffer", "dv2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "kcp_segments_heads"},
        "kernel_symbol": "kernel_cake_kda_train_3808ea4d567b82ea21a4",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "use_pdl": False,
        },
        "library": "cake_kda_train_3808ea4d567b82ea21a4",
        "logical": {
            "dh_tape": "dh",
            "dht": "dht",
            "dht_hg": "dht_hg",
            "dht_index": "dht_index",
            "dht_n": "dht_n",
            "do_tma": "do_int",
            "dv1": "dv1",
            "dv2": "dv2",
            "gk": "gk",
            "kg_tma": "kg",
            "num_heads": "num_heads",
            "num_seqs": "num_seqs",
            "out": "out",
            "out_hg": "out_hg",
            "out_index": "out_index",
            "out_n": "out_n",
            "qg_tma": "qg",
            "scale": "scale",
            "seq_chunk_start": "seg_chunk_start",
            "tile_stride": "tile_stride",
            "w_tma": "w",
            "walk_seg": "walk_seg",
        },
        "program": "kcp_dhu_tape",
        "select_rule": {"kind": "always"},
        "test_knobs": {"walk_form": "full"},
        "variant": "full",
    },
}


# ----------------------------------------------------------------------------------------------------------------
# Selection and grids


def records_of_program(program: str) -> tuple[str, ...]:
    """The record names of ``program`` in registry order."""
    if program not in PROGRAMS:
        raise ValueError(
            f"unknown KCP program {program!r}; the programs are {PROGRAMS}"
        )
    return tuple(name for name, rec in MODULES.items() if rec["program"] == program)


def _admits(
    rule: Mapping[str, Any], *, ncols: int | None, form: str | None = None
) -> bool:
    kind = rule["kind"]
    if kind == "always":
        return True
    if kind == "columns":
        return ncols is not None and int(ncols) == int(rule["ncols"])
    if kind == "walk_form":
        return form is not None and str(form) == str(rule["form"])
    raise ValueError(f"unknown Cake KDA KCP selection rule: {kind!r}")


def select_record(
    program: str, *, ncols: int | None = None, form: str | None = None
) -> str:
    """The one record of ``program`` the launch's scalars select.

    ``always`` is the rule of a single-record program; ``columns`` (``{"kind":
    "columns", "ncols": N}``) picks the merge variant whose column count is
    the chain's ``ncols`` (128 for state chains, 256 for map chains);
    ``walk_form`` (``{"kind": "walk_form", "form": F}``) picks the K1 / K2
    walk record of form ``F`` -- ``"full"`` (one CTA per segment and head) or
    ``"split"`` (the two-CTA record over 64 value columns each), the
    launch's form from :func:`walk_form`; a program under a fixed rule ships
    its one ``always`` record, the other form being test-only
    (:func:`test_walk_record`).  Exactly one record admits every launch, or
    the registry is inconsistent.
    """
    admitted = [
        name
        for name in records_of_program(program)
        if _admits(MODULES[name]["select_rule"], ncols=ncols, form=form)
    ]
    if len(admitted) != 1:
        raise ValueError(
            f"program {program!r}: {len(admitted)} records admit ncols={ncols} form={form} ({admitted}); expected one"
        )
    return admitted[0]


def walk_form(
    program: str,
    chains: Sequence[int],
    num_heads: int,
    sm_count: int,
    *,
    knob: str = "auto",
) -> str:
    """The form of one launch of walk ``program``: ``"full"`` (one CTA per
    segment and head over the 128 value columns) or ``"split"`` (the two-CTA
    record: two CTAs per segment and head over 64 value columns each; bitwise
    equal).  ``chains`` are the chunk counts of the listed segments, ``knob``
    the ``walk_form`` test knob (``"full"`` / ``"split"`` force a form).

    Under the rule ``fit`` (:data:`WALK_FORM_RULES`, the offset programs) a
    launch takes the split form when its ``2 * num_heads`` CTAs per chain, two
    resident per SM, list-schedule (LPT, :func:`cake_kcp_plan.lpt_makespan`)
    within the longest chain, counting a two-CTA step at
    :data:`WALK_FORM_FIT_STEP_RATIO` of a full-width step: there every CTA runs
    beside another one that hides its per-step latency, and the launch lasts
    one longest chain of shorter steps.  A launch of more work than that runs
    several waves, where the full-width CTA does more per issued instruction
    and keeps the SMs busier.  Under the rule ``split`` (the tape programs,
    whose per-step tape stores halve with the CTA's columns) every launch takes
    the split form.  An empty list is ``"full"``.
    """
    if knob not in WALK_FORMS:
        raise ValueError(f"walk_form must be one of {WALK_FORMS}, got {knob!r}")
    if knob != "auto":
        return knob
    rule = WALK_FORM_RULES[program]
    if rule != "fit":
        return rule
    steps = [int(c) for c in chains if int(c) > 0]
    if not steps:
        return "full"
    span = cake_kcp_plan.lpt_makespan(
        [(c, 2 * int(num_heads)) for c in steps], 2 * int(sm_count)
    )
    return "split" if span * WALK_FORM_FIT_STEP_RATIO <= max(steps) else "full"


def launch_grid(
    name: str,
    *,
    n_walk: int = 0,
    num_heads: int = 0,
    n_chains: int = 0,
    ncols: int = 0,
    num_chunks_pad: int = 0,
    sm_count: int = 0,
) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule.

    ``kcp_segments_heads`` launches ``(n_walk * num_heads * ctas_per_head, 1, 1)``
    (one CTA per listed segment and head -- ``ctas_per_head`` 2 for the
    half-width walks: the BF16x3 transfer walk and the two-CTA state walks;
    the host lists the segments longest chain first);
    ``kcp_chains`` launches ``(n_chains * (ncols / cols_per_cta), 1, 1)`` (one
    CTA per chain and column block; ``ncols`` is a multiple of the record's
    ``cols_per_cta``); ``kcp_sm_bounded`` the persistent grid ``(max(1,
    min(num_chunks_pad * num_heads, ctas_per_sm * sm_count)), 1, 1)`` over the
    (chunk, head) tiles of the rank's padded chunk count; ``kcp_pairs``
    ``(num_chunks_pad / 2, num_heads, 1)`` (one CTA per chunk pair and head;
    ``num_chunks_pad`` is even).
    """
    record = MODULES.get(name) or TEST_MODULES.get(name)
    if record is None:
        raise KeyError(f"{name!r} is not a KCP record")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "kcp_segments_heads":
        if int(n_walk) < 1 or int(num_heads) < 1:
            raise ValueError(
                f"record {name}: n_walk={n_walk} and num_heads={num_heads} must be >= 1"
            )
        return (int(n_walk) * int(num_heads) * int(rule.get("ctas_per_head", 1)), 1, 1)
    if kind == "kcp_chains":
        cols = int(rule["cols_per_cta"])
        if int(n_chains) < 1 or int(ncols) < 1 or int(ncols) % cols:
            raise ValueError(
                f"record {name}: n_chains={n_chains} must be >= 1 and ncols={ncols} a positive multiple of {cols}"
            )
        return (int(n_chains) * (int(ncols) // cols), 1, 1)
    if kind == "kcp_sm_bounded":
        if int(num_chunks_pad) < 1 or int(num_heads) < 1 or int(sm_count) < 1:
            raise ValueError(
                f"record {name}: num_chunks_pad={num_chunks_pad}, num_heads={num_heads} and "
                f"sm_count={sm_count} must be >= 1"
            )
        bound = int(rule["ctas_per_sm"]) * int(sm_count)
        return (max(1, min(int(num_chunks_pad) * int(num_heads), bound)), 1, 1)
    if kind == "kcp_pairs":
        if int(num_chunks_pad) < 1 or int(num_heads) < 1 or int(num_chunks_pad) % 2:
            raise ValueError(
                f"record {name}: num_chunks_pad={num_chunks_pad} must be a positive even count and "
                f"num_heads={num_heads} >= 1"
            )
        return (int(num_chunks_pad) // 2, int(num_heads), 1)
    raise ValueError(f"unknown Cake KDA KCP grid rule: {kind!r}")


# ----------------------------------------------------------------------------------------------------------------
# Loading and launching (``build.json``)

arch_for_capability = cake_jit.arch_for_capability
device_arch = cake_jit.device_arch
sm_count = cake_jit.sm_count


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake KDA KCP record {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a
    builder compiled it, else FlashInfer's JIT build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of the package's ``build.json`` without a prebuilt file (the
    chunk stages' and the KCP programs' alike; for ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter and ``(None, axis)`` per grid argument, in
    ``run`` order; computed once."""
    record = MODULES.get(name) or TEST_MODULES[name]
    logical = record["logical"]
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


_ARCH_BY_INDEX: dict[int, str] = {}
_FFI_DEVICES: dict[int, Any] = {}


def _arch_of(device_index: int) -> str:
    """:func:`device_arch` cached per device index (no capability query per launch)."""
    arch = _ARCH_BY_INDEX.get(int(device_index))
    if arch is None:
        arch = _ARCH_BY_INDEX[int(device_index)] = device_arch(int(device_index))
    return arch


def _ffi_stream(device_index: int) -> Any:
    """Trimmed host path: the FFI stream context of a trimmed launch -- the device's current torch stream as a raw handle
    (``use_raw_stream``), the stream ``tvm_ffi.use_torch_stream()`` would set without building a torch.cuda.Stream."""
    import torch  # noqa: PLC0415

    dev = _FFI_DEVICES.get(int(device_index))
    if dev is None:
        dev = _FFI_DEVICES[int(device_index)] = tvm_ffi.device(
            f"cuda:{int(device_index)}"
        )
    get = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = (
        int(get(int(device_index)))
        if get is not None
        else int(torch.cuda.current_stream(int(device_index)).cuda_stream)
    )
    return tvm_ffi.use_raw_stream(dev, raw)


def run_module(
    name: str,
    values: Mapping[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
    module: Any | None = None,
    trim: bool = False,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to tensors and Python scalars; the
    record maps its kernel parameters onto that vocabulary and ``grid`` is the plan's launch grid
    (:func:`launch_grid`).  ``module`` overrides the loaded library (a test-only build, :func:`load_test`); by
    default the record's library of the device's architecture runs.  The kernel runs on the current torch stream.
    """
    record = MODULES.get(name) or TEST_MODULES[name]
    args = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    if trim:  # same module, same arguments, same stream; the arch and the FFI device are cached
        if module is None:
            module = load(name, _arch_of(device_index))
        with _ffi_stream(device_index):
            getattr(module, record["ffi_entry"])(*args)
        return
    if module is None:
        module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


# ----------------------------------------------------------------------------------------------------------------
# The test-only record set (``build.test.json``)

_PACKAGE = Path(__file__).resolve().parent
_TEST_MANIFEST_PATH = _PACKAGE / TEST_MANIFEST
_MANIFEST_SCHEMA = "cake-build/1"


def test_knobs_enabled() -> bool:
    """Whether the process opted into the test-only programs (:func:`cake_kcp_plan.enable_test_knobs`)."""
    return cake_kcp_plan.test_knobs_enabled()


def test_record(
    precision_mode: int = 0, transfer_walk: str = "bf16x3_hw64"
) -> tuple[str, dict[str, Any]] | None:
    """``(record name, record)`` of the test-only transfer walk the knobs select, ``None`` for the shipped program
    (``precision_mode`` 0 and ``transfer_walk`` ``"bf16x3_hw64"``).  ``transfer_walk`` ``"tf32_hilo"`` is the TF32
    hi/lo walk and ``"bf16x3"`` the full-width BF16x3 form; ``precision_mode`` 1 / 2 are the static programs of the
    TF32 hi/lo walk (``transfer_walk`` left at its default or ``"tf32_hilo"``); a BF16x3 walk beside a non-zero
    ``precision_mode`` is refused (one test-only program per launch)."""
    mode, walk = int(precision_mode), str(transfer_walk)
    if mode not in PRECISION_MODES:
        raise ValueError(
            f"precision_mode must be one of {PRECISION_MODES}, got {precision_mode!r}"
        )
    if walk not in TRANSFER_WALKS:
        raise ValueError(
            f"transfer_walk must be one of {TRANSFER_WALKS}, got {transfer_walk!r}"
        )
    if mode == 0 and walk == TRANSFER_WALKS[0]:
        return None
    if mode != 0 and walk not in (TRANSFER_WALKS[0], "tf32_hilo"):
        raise ValueError(
            f"precision_mode {mode} and transfer_walk {walk!r} name two test-only programs; set one knob"
        )
    wanted: dict[str, int | str] = (
        {"precision_mode": mode} if mode != 0 else {"transfer_walk": walk}
    )
    for name, record in TEST_MODULES.items():
        if record["test_knobs"] == wanted:
            return name, record
    raise ValueError(f"no test-only record carries the knobs {wanted}")


def test_walk_record(program: str, form: str) -> tuple[str, dict[str, Any]]:
    """``(record name, record)`` of the test-only form ``form`` of walk ``program`` under a fixed rule of
    :data:`WALK_FORM_RULES` (the full-width tape walks beside their shipped two-CTA records; the ``walk_form``
    knob alone selects them)."""
    wanted = {"walk_form": str(form)}
    for name, record in TEST_MODULES.items():
        if record["program"] == program and record.get("test_knobs") == wanted:
            return name, record
    raise ValueError(f"no test-only record of {program!r} carries the knobs {wanted}")


def _require_test_knobs() -> None:
    if not test_knobs_enabled():
        raise RuntimeError(
            f"the test-only KCP programs are loaded only after {cake_kcp_plan.ENABLE_TEST_KNOBS}; the shipped "
            "route (precision_mode 0) uses build.json alone"
        )


def test_manifest() -> dict[str, Any]:
    """The parsed ``build.test.json`` (schema ``cake-build/1``, the test-only libraries), read on first use and
    only after :func:`cake_kcp_plan.enable_test_knobs`: the shipped route never opens it.  The opt-in is checked on
    every call, the parse is cached (an opt-out after a load closes it again)."""
    _require_test_knobs()
    return _test_manifest()


@functools.cache
def _test_manifest() -> dict[str, Any]:
    try:
        text = _TEST_MANIFEST_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{_TEST_MANIFEST_PATH} is missing: the test-only record set was not exported"
        ) from exc
    document = json.loads(text)
    if document.get("schema") != _MANIFEST_SCHEMA:
        raise RuntimeError(
            f"{_TEST_MANIFEST_PATH} has schema {document.get('schema')!r}; expected {_MANIFEST_SCHEMA!r}"
        )
    if document.get("package") != cake_build.manifest()["package"]:
        raise RuntimeError(
            f"{_TEST_MANIFEST_PATH} names package {document.get('package')!r}, not this package"
        )
    return document


def test_library_name(name: str, arch: str) -> str:
    """``build.test.json`` library of test-only record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = TEST_MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"test-only record {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


def _test_entry(library: str) -> dict[str, Any]:
    entries = {entry["name"]: entry for entry in test_manifest()["libraries"]}
    try:
        return entries[library]
    except KeyError:
        raise KeyError(
            f"{library!r} is not a library of {_TEST_MANIFEST_PATH}"
        ) from None


def test_jit_spec(library: str) -> Any:
    """FlashInfer's JIT spec of a ``build.test.json`` library, derived exactly as ``cake_build.jit_spec`` derives
    one from ``build.json``: digest-verified sources, ``arch_flags[arch]`` + the entry's ``nvcc_flags`` + the
    ``nvcc_flags_if`` items the local toolkit matches, the closure digest in the spec name."""
    document = test_manifest()
    entry = _test_entry(library)
    gen_jit_spec, get_cuda_version = cake_build._jit_toolchain(library)
    cuda = cake_build._local_cuda_version(get_cuda_version)
    if not cuda >= cake_build._version(entry["min_cuda"]):
        raise RuntimeError(
            f"{library} needs CUDA {entry['min_cuda']} or newer; the local toolkit is {cuda[0]}.{cuda[1]}"
        )
    sources: list[Path] = []
    for relative in entry["sources"]:
        path = _PACKAGE / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"][relative]:
            raise RuntimeError(
                f"{path} does not match its {TEST_MANIFEST} digest for {library}: {digest} != {entry['sha256'][relative]}"
            )
        sources.append(path)
    flags = [
        *document["arch_flags"][entry["arch"]],
        *entry["nvcc_flags"],
        *cake_build._conditional_flags(entry["nvcc_flags_if"], cuda),
    ]
    return gen_jit_spec(
        f"{library}_{entry['closure_sha256'][:20]}",
        sources,
        extra_cuda_cflags=flags,
        extra_ldflags=list(document["common"]["link_flags"]),
        extra_include_paths=[_PACKAGE / d for d in document["common"]["include_dirs"]],
        use_fast_math=False,
    )


def load_test(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of test-only record ``name`` (a key of ``TEST_MODULES``) built for ``arch``,
    JIT-built from ``build.test.json`` (never prebuilt, never from ``build.json``); the opt-in is checked on every
    call, the loaded module is cached."""
    _require_test_knobs()
    return _load_test(name, arch)


@functools.cache
def _load_test(name: str, arch: str) -> Any:
    return test_jit_spec(test_library_name(name, arch)).build_and_load()


def test_specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of the ``arch`` libraries of ``build.test.json`` (test warm-up)."""
    return [
        test_jit_spec(entry["name"])
        for entry in test_manifest()["libraries"]
        if entry["arch"] == arch
    ]


# ----------------------------------------------------------------------------------------------------------------
# The composition's launches (``cake_kcp.KernelBackend.run`` -> :func:`run_program`)

# Per walk program: the compact pool its initial state / final adjoint comes from (``None``: none) and the pool it
# writes, by the argument names of ``cake_kcp.KcpCall.walk`` (= the records' logical names).
_WALK_POOLS: dict[str, tuple[str | None, str]] = {
    "kcp_fwdh_offset": ("h0", "out"),
    "kcp_fwdh_tape": ("h0", "out"),
    "kcp_dhu_offset": ("dht", "out"),
    "kcp_dhu_tape": ("dht", "out"),
    "kcp_transfer_walk": (None, "out"),
}
# Per walk program: the local stage values it reads (the logical names of ``cake_backend``'s value dict, which the
# composition holds under ``KcpCall.local_values``): the prep stages' operands and the tapes.
_WALK_OPERANDS: dict[str, tuple[str, ...]] = {
    "kcp_fwdh_offset": ("w", "kg", "u", "gk", "h", "v_new"),
    "kcp_fwdh_tape": ("w", "kg", "u", "gk", "h", "v_new"),
    "kcp_dhu_offset": ("w", "kg", "qg", "do_int", "dv1", "gk", "dh", "dv2"),
    "kcp_dhu_tape": ("w", "kg", "qg", "do_int", "dv1", "gk", "dh", "dv2"),
    "kcp_transfer_walk": ("w", "kg", "u", "gk"),
}


@functools.lru_cache(maxsize=4096)
def _device_table(values: tuple[int, ...], dtype_name: str, device: str) -> Any:
    """A device copy of a host table (cached: the tables of a ``(ctx, H)`` problem are reused call after call)."""
    import torch

    return torch.tensor(list(values), dtype=getattr(torch, dtype_name), device=device)


@functools.lru_cache(maxsize=None)
def _dummy_tile(device: str) -> Any:
    """One zero FP32 tile per device: the valid buffer a launch passes for a pool it never reads or writes."""
    import torch

    return torch.zeros(TILE, dtype=torch.float32, device=device)


def _flat_pool(pool: Any, what: str) -> Any:
    import torch

    if not isinstance(pool, torch.Tensor):
        raise TypeError(f"{what}: expected a tensor pool, got {type(pool).__name__}")
    if pool.dtype != torch.float32 or not pool.is_cuda:
        raise ValueError(
            f"{what}: a state pool is an FP32 CUDA tensor, got {pool.dtype} on {pool.device}"
        )
    if pool.dim() != 1 or not pool.is_contiguous():
        raise ValueError(
            f"{what}: a state pool is a flat contiguous tensor, got shape {tuple(pool.shape)}"
        )
    return pool


def _pool_arguments(
    ix: Any, *, what: str, H: int, nseg: int, device: str, trim: bool = False
) -> tuple[Any, Any, int, int, int | None]:
    """``(pool view, slot table, hg, n, tile_stride)`` of a compact pool as the kernels address it.

    ``ix`` is a ``cake_kcp.Indexed`` (pool, ``index[nseg]``, ``tiles[slot][h]``): its tile table is recovered into
    the two-integer form through ``Indexed.addressing`` and the pool view starts at its base tile.  ``None`` (or a
    non-pool value, the tape program's unused ``out``) is the absent pool: every slot ``-1`` over a dummy tile, with
    no stride of its own.
    """
    if ix is None or not hasattr(ix, "addressing"):
        return (
            _dummy_tile(device),
            _device_table((-1,) * nseg, "int32", device),
            int(H),
            1,
            None,
        )
    meta = getattr(ix, "meta", None) if trim else None
    H = int(H)
    if meta is not None and meta.H == H and meta.device == device:
        # the tables' precomputed addressing and device index table (cake_kcp._Tables.indexed_meta)
        base_tile, hg, n, stride = meta.addr
        if len(meta.index) != nseg:
            raise ValueError(
                f"{what}: the slot table has {len(meta.index)} entries, the problem {nseg} segments"
            )
        pool = _flat_pool(ix.pool, what)
        view = pool[int(base_tile) * TILE :] if int(base_tile) else pool
        return view, meta.dev_index, int(hg), int(n), int(stride)
    base_tile, hg, n, stride = ix.addressing(H)
    index = tuple(int(i) for i in ix.index)
    if len(index) != nseg:
        raise ValueError(
            f"{what}: the slot table has {len(index)} entries, the problem {nseg} segments"
        )
    pool = _flat_pool(ix.pool, what)
    view = pool[int(base_tile) * TILE :] if int(base_tile) else pool
    return view, _device_table(index, "int32", device), int(hg), int(n), int(stride)


def _local_values(call: Any) -> Mapping[str, Any]:
    local = getattr(call, "local_values", None)
    if local is None:
        raise RuntimeError(
            "the KCP kernel backend needs the call's local stage values (KcpCall.local_values: the logical value "
            "dict of the chunk stages, cake_backend's vocabulary) before the first walk launch"
        )
    return local


def _walk(name: str, args: Mapping[str, Any], call: Any, device_index: int) -> None:
    trim = int(getattr(call, "host_trim", 0)) == 1
    walk_list = args["walk"]
    n_walk = len(walk_list)
    if not n_walk:
        return
    local = _local_values(call)
    H, nseg = int(call.H), int(call.tb.NSEG)
    device = str(call.device)
    values: dict[str, Any] = {}
    for key in _WALK_OPERANDS[name]:
        if key not in local:
            raise RuntimeError(
                f"{name}: the local stage values carry no {key!r} (needed by the walk)"
            )
        values[key] = local[key]
    if (
        trim
    ):  # the table set's identity-cached device copies of its own lists (same integers)
        values.update(
            num_heads=H,
            num_seqs=nseg,
            seg_chunk_start=call.tb.dev_list(call.tb.hp.seg_chunk_start, device),
            walk_seg=call.tb.dev_list(walk_list, device),
        )
    else:
        walk = tuple(int(s) for s in walk_list)
        values.update(
            num_heads=H,
            num_seqs=nseg,
            seg_chunk_start=_device_table(
                tuple(int(x) for x in call.tb.hp.seg_chunk_start), "int32", device
            ),
            walk_seg=_device_table(walk, "int32", device),
        )
    in_pool, out_pool = _WALK_POOLS[name]
    stride: int | None = None
    for key in (in_pool, out_pool):
        if key is None:
            continue
        pool, index, hg, n, pool_stride = _pool_arguments(
            args.get(key),
            what=f"{name}.{key}",
            H=H,
            nseg=nseg,
            device=device,
            trim=trim,
        )
        values.update(
            {key: pool, f"{key}_index": index, f"{key}_hg": hg, f"{key}_n": n}
        )
        if pool_stride is not None:
            if stride is not None and stride != pool_stride:
                raise ValueError(
                    f"{name}: the pools of one launch must share one tile_stride ({stride} vs {pool_stride})"
                )
            stride = pool_stride
    values["tile_stride"] = 1 if stride is None else stride
    if name in ("kcp_dhu_offset", "kcp_dhu_tape"):
        values["scale"] = float(call.scale)
    if name == "kcp_transfer_walk":
        values.update(
            init_mode=int(args["init_mode"]),
            use_u=int(args["use_u"]),
            write_transposed=int(args["write_transposed"]),
        )
    module = None
    if name in WALK_FORM_RULES:  # the K1 / K2 walks: the form of this launch
        form = str(args.get("walk_form") or "")
        if form not in ("full", "split"):
            seg = call.tb.hp.seg_chunk_start
            form = walk_form(
                name,
                [int(seg[s + 1]) - int(seg[s]) for s in walk_list],
                H,
                sm_count(device_index),
                knob=str(_call_knob(call, "walk_form", "auto")),
            )
        rule = WALK_FORM_RULES[name]
        if rule in ("full", "split") and form != rule:
            # the other form of a fixed-rule walk: its test-only record (the knob alone selects it)
            record = test_walk_record(name, form)[0]
            module = load_test(record, device_arch(device_index))
        else:
            record = select_record(name, form=form)
        WALK_FORM_STATS[form] += 1
        log = getattr(call, "walk_form_log", None)
        if log is not None:
            log.append((name, form, n_walk))
    else:
        record = select_record(name)
    if name == "kcp_transfer_walk":
        chosen = test_record(
            _call_knob(call, "precision_mode", 0),
            _call_knob(call, "transfer_walk", TRANSFER_WALKS[0]),
        )
        if chosen is not None:
            record = chosen[0]
            module = load_test(record, device_arch(device_index))
    run_module(
        record,
        values,
        grid=launch_grid(record, n_walk=n_walk, num_heads=H),
        device_index=device_index,
        module=module,
        trim=trim,
    )


def _call_knob(call: Any, knob: str, default: Any) -> Any:
    """A test knob of the call: ``KcpCall.<knob>``, else ``KcpCall.opts.<knob>``, else ``default``."""
    value = getattr(call, knob, None)
    if value is None:
        value = getattr(getattr(call, "opts", None), knob, None)
    return default if value is None else value


def _chain_tables(chains: Any, device: str) -> dict[str, Any]:
    key = (
        tuple(int(x) for x in chains.chain_start),
        tuple(int(x) for x in chains.elem_m),
        tuple(int(x) for x in chains.elem_c),
        tuple(int(x) for x in chains.elem_y),
        tuple(int(x) for x in chains.x0_index),
        tuple(int(x) for x in chains.y_final),
    )
    return _chain_tables_cached(key, device)


@functools.lru_cache(maxsize=256)
def _chain_tables_cached(
    key: tuple[tuple[int, ...], ...], device: str
) -> dict[str, Any]:
    chain_start, elem_m, elem_c, elem_y, x0_index, y_final = key
    # an element table of an all-empty chain set still needs a valid pointer; the kernel reads no element then
    pad = lambda table: table if table else (0,)  # noqa: E731
    return {
        "chain_start": _device_table(chain_start, "int32", device),
        "elem_m": _device_table(pad(elem_m), "int64", device),
        "elem_c": _device_table(pad(elem_c), "int64", device),
        "elem_y": _device_table(pad(elem_y), "int64", device),
        "x0_index": _device_table(x0_index, "int64", device),
        "y_final": _device_table(y_final, "int64", device),
    }


def _merge(args: Mapping[str, Any], call: Any, device_index: int) -> None:
    chains = args["chains"]
    n_chains = int(chains.n_chains)
    if n_chains == 0:
        return
    device = str(call.device)
    ncols = int(chains.ncols)
    record = select_record("kcp_affine_merge", ncols=ncols)
    dummy = _dummy_tile(device)
    values: dict[str, Any] = {
        "m": _flat_pool(args["m"], "kcp_affine_merge.m"),
        "c": dummy
        if args.get("c") is None
        else _flat_pool(args["c"], "kcp_affine_merge.c"),
        "x0": dummy
        if args.get("x0") is None
        else _flat_pool(args["x0"], "kcp_affine_merge.x0"),
        "y": _flat_pool(args["y"], "kcp_affine_merge.y"),
        "ncols": ncols,
        "transpose_m": int(args["transpose_m"]),
        "write_states": int(args["write_states"]),
        "cols_per_cta": int(MODULES[record]["grid_rule"]["cols_per_cta"]),
    }
    trim = int(getattr(call, "host_trim", 0)) == 1
    if trim and hasattr(
        chains, "device_tables"
    ):  # the chain set's own cached device tables (same integers)
        values.update(chains.device_tables(device))
    else:
        values.update(_chain_tables(chains, device))
    run_module(
        record,
        values,
        grid=launch_grid(record, n_chains=n_chains, ncols=ncols),
        device_index=device_index,
        trim=trim,
    )


def run_program(name: str, args: Mapping[str, Any], *, device_index: int) -> None:
    """Launch KCP program ``name`` (a walk or the merge of :data:`PROGRAMS`) with the composition's arguments
    ``args``.  The chunk stages ``kcp_fwd_o`` / ``kcp_dav`` take no composition arguments: the host launches
    them through :func:`run_module` with their logical values on the grid of :func:`launch_grid`.

    ``args`` is the vocabulary of ``cake_kcp.Backend.run``: for a walk the segment list ``walk`` (a launch over no
    segment is skipped), the compact pools ``h0`` / ``dht`` / ``out`` as ``Indexed`` (or ``None``), the transfer
    walk's ``init_mode`` / ``use_u`` / ``write_transposed``; for the merge the tile pools ``m`` / ``c`` / ``x0`` /
    ``y`` (``c`` / ``x0`` may be ``None``), the ``chains`` tables and ``transpose_m`` / ``write_states``; and
    ``_call``, the ``KcpCall`` of the launch, which supplies the head count, the segment table, the operand and
    tape tensors (``local_values``), ``scale`` and ``precision_mode``.  The record is selected from the launch's
    scalars (:func:`select_record`), the grid is its rule's (:func:`launch_grid`), and the kernel runs on the
    current torch stream of the device.
    """
    call = args.get("_call")
    if call is None:
        raise ValueError(
            "run_program needs the KcpCall of the launch under args['_call']"
        )
    if name == "kcp_affine_merge":
        _merge(args, call, device_index)
    elif name in WALK_PROGRAMS:
        _walk(name, args, call, device_index)
    elif name in PROGRAMS:
        raise ValueError(
            f"KCP program {name!r} is a chunk stage: launch it through run_module with its logical values"
        )
    else:
        raise ValueError(f"unknown KCP program {name!r}; the programs are {PROGRAMS}")


__all__ = [
    "ARCHES",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PRECISION_MODES",
    "PROGRAMS",
    "TEST_MANIFEST",
    "TEST_MODULES",
    "TILE",
    "TRANSFER_WALKS",
    "WALK_FORMS",
    "WALK_FORM_FIT_STEP_RATIO",
    "WALK_FORM_RULES",
    "WALK_FORM_STATS",
    "WALK_PROGRAMS",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "load_test",
    "records_of_program",
    "run_module",
    "run_program",
    "select_record",
    "sm_count",
    "specs_for_arch",
    "test_jit_spec",
    "test_knobs_enabled",
    "test_library_name",
    "test_manifest",
    "test_record",
    "test_specs_for_arch",
    "test_walk_record",
    "walk_form",
]
