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

Host plan and loader glue of the Cake context-parallel (KCP) packed width-4
depthwise causal convolution + SiLU training kernels of the non-deterministic
package (SM100 / SM103 / SM107): the halo-aware programs the KCP host flow
(:mod:`.kcp`) launches, recorded beside the zero-history registry
:mod:`.cake_jit`, whose tables this module never touches.

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only, in the
vocabulary of :mod:`.cake_jit`: the kernel's ``library`` stem, the
architectures it is delivered for, its launch record, grid rule, workspace
rule, selection rule and the kernel-parameter -> logical-argument map.  What to
compile -- sources, digests, nvcc flags -- is recorded once in ``build.json``
next to this file (its entries sit beside the zero-history entries);
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's JIT
from the same sources and flags, verifying every source digest first.

The routes of an architecture are the three KCP kinds of the det package
(``flashinfer.cake_causal_conv1d_det_train_kcp``): ``kcp_fwd``
(this package's copy of the det KCP forward: ``y`` and the fragments' history
rows ``hist``), ``kcp_bwd_head`` (this package's copy of the det head program:
the head-row derivatives ``dpre_head`` every consumer fragment sends back;
grid :func:`head_grid`) and ``kcp_bwd`` (the fallback program and the
big-channel ``_st8`` program, selected by the det channel rule).  Neither copy
has a cross-CTA reduction, so their outputs are the det programs' bits.  The
``kcp_bwd`` programs are the det KCP backward bodies with the accumulator
reduction of this package: ONE ORDINARY launch (``occupancy`` grid of
``ctas_per_sm x SM count`` CTAs, no cooperative attribute, no grid barrier)
writes ``dx`` (the received ``dpre_tail`` rows folded in before the one BF16
rounding: the det program's bits) and the rank-local FP32 ``dweight``
partial, which every CTA adds into the FP32 accumulator ``acc`` (``[C, 4]``)
before its channel tile's last CTA (a ticket on ``work``, uint32
``[n_ch_tiles]``) stores ``dw`` and zeroes the tile's accumulators and
counter.  So ``acc`` / ``work`` are zero before and after every launch; only
the run-to-run bits of ``dweight`` are relaxed.

Two more routes carry the overlapped backward the host runs for up to
:data:`DYN_MAX_CHANNELS` channels (:data:`ROUTE_KINDS`; :data:`KINDS` stays the
det package's three): ``kcp_bwd_dyn``, ONE ORDINARY launch of
:func:`dyn_plan`'s ``n_ctas`` CTAs that runs while the halo exchange is in
flight, every ``tail_row_off`` entry ``-1`` (``dx`` of every row, a fragment's
last rows as at a document end, and the rank's FP32 ``dweight`` partial, whose
CTAs take their units of tokens from lanes, from the back of other lanes and
from the CTAs they cancel through Cluster Launch Control, and add their runs
into ``DW_COPIES`` (the rule's ``slots``) FP32 copies of the accumulators
``acc`` before each channel tile's last warp sums the copies, stores ``dw`` and
zeroes them; its ``skip_row_off`` table names the fragments whose last
``min(3, len)`` rows it leaves unwritten: none with an all ``-1`` table, the
fragments with a tail row with ``tail_row_off``), and ``kcp_bwd_halo``,
launched once the exchanged ``dpre_tail`` rows arrived (grid
:func:`head_grid`), which writes the last ``min(3, len)`` rows of every
fragment with a tail row with the tail rows folded in, after the main program
on its stream or beside it on another when the main program leaves those
rows, so ``dx`` is the det program's bits on every row.  The main program's ``work``
words are the tiles' warp words ``tile_tk`` (``DYN_TILE_WORDS`` per tile of
:data:`DYN_TILE` channels), the claim words ``claims`` (``L``, one per lane)
and the departure word ``departs`` (one), all zero before and after every
launch (:func:`dyn_workspace_layout`).

One more route carries the two-launch forward's history-row program
(:data:`FWD_HALO_KIND`, ``kcp_fwd_halo``: this package's copy of the det
program of the det package's route of that name).  The host can run the
forward as a main program that reads no history row -- the zero-history
registry's single-rank forward program on the rank's fragments, or the
``kcp_fwd`` program with every ``hist_row_off`` entry ``-1`` -- launched before
or after the forward halo exchange is waited; after the wait this program
writes ``y`` of the first ``min(3, len)`` rows of every fragment with a history
row again from the received ``hist`` rows, with the ``kcp_fwd`` program's
instructions in its order, so ``y`` is the one-launch forward's bits.  Its grid
is the head program's rule, :func:`head_grid`.

The host plan reuses :mod:`.cake_jit`'s functions over a record (:func:`plan`,
:func:`launch_grid`, :func:`accumulator_layout`, :func:`channel_split`)
unchanged; the functions bound to the tables (:func:`route_modules`,
:func:`selected_modules`, :func:`workspace_record`, :func:`library_name`,
:func:`load`, :func:`run_module`) are spelled here over ``MODULES`` and
``ROUTES``.  The logical tensor names are :mod:`.cake_jit`'s plus the KCP
tables ``hist``, ``hist_row_off``, ``dpre_tail``, ``tail_row_off``,
``skip_row_off``, ``dpre_head`` and ``head_row_off``; the two ``CUtensorMap`` parameters
``x_map`` / ``dy_map`` of the TMA backward programs map onto ``x`` / ``dy``
(the binding encodes the descriptors from the tensors at every launch); the
scalars are :mod:`.cake_jit`'s.

The program names are ``cake_causal_conv1d_train_<20 hex>_kcp``: the copies of
the det forward, history-row and head programs take the det programs' identity
digits under this package's prefix, the backward programs the identity digests
of their own kernel module key.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build
from . import cake_jit as _zero_history
from .cake_jit import (  # noqa: F401  (re-exported: the host plan over a record is the zero-history one)
    ARCH_BY_CAPABILITY,
    ARCHES,
    CHANNEL_MULTIPLE,
    INT32_MAX,
    SELECT_KINDS,
    WIDTH,
    accumulator_layout,
    arch_for_capability,
    channel_split,
    launch_grid,
    plan,
    sm_count,
)

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    # Route ``kcp_fwd``: this package's copy of the det program
    # ``cake_causal_conv1d_det_train_675940a7a4fc0dba4b53_kcp`` (the same build keywords and kernel under this
    # package's prefix; no cross-CTA reduction, so its outputs are bitwise the det program's).
    "cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "kcp_fwd",
        "workspace_rule": None,
    },
    # Route ``kcp_bwd``: the KCP backward body of the det program
    # ``cake_causal_conv1d_det_train_5288d132a5cc927ae729_kcp`` (the same build keywords and channel rule) whose
    # cross-CTA dweight reduction is the accumulator form: an ORDINARY launch (no cooperative attribute, no grid
    # barrier) of ``ctas_per_sm x SM count`` CTAs, each adding its FP32 CTA row into ``acc`` (FP32 ``[C x 4]``,
    # ``red.global.add`` v4) and taking a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``); the tile's last CTA stores the rank-local FP32 ``dw`` and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  ``dx`` (with the folded
    # ``dpre_tail`` rows) is bitwise the det program's.
    "cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "kcp_bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "slots": 1,
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``kcp_bwd``: the KCP backward body of the det program
    # ``cake_causal_conv1d_det_train_6ea6d5fc9e7fa3f2d039_kcp`` (the same build keywords and channel rule) whose
    # cross-CTA dweight reduction is the accumulator form: an ORDINARY launch (no cooperative attribute, no grid
    # barrier) of ``ctas_per_sm x SM count`` CTAs, each adding its FP32 CTA row into ``acc`` (FP32 ``[C x 4]``,
    # ``red.global.add`` v4) and taking a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``); the tile's last CTA stores the rank-local FP32 ``dw`` and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  ``dx`` (with the folded
    # ``dpre_tail`` rows) is bitwise the det program's.
    "cake_causal_conv1d_train_777e24f8771c3f7abdf8_kcp": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_777e24f8771c3f7abdf8_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_777e24f8771c3f7abdf8_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_777e24f8771c3f7abdf8_kcp",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "kcp_bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "slots": 1,
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``kcp_bwd``: the KCP backward body of the det program
    # ``cake_causal_conv1d_det_train_7485529a1755e5bd3036_kcp`` (the same build keywords and channel rule) whose
    # cross-CTA dweight reduction is the accumulator form: an ORDINARY launch (no cooperative attribute, no grid
    # barrier) of ``ctas_per_sm x SM count`` CTAs, each adding its FP32 CTA row into ``acc`` (FP32 ``[C x 4]``,
    # ``red.global.add`` v4) and taking a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``); the tile's last CTA stores the rank-local FP32 ``dw`` and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  ``dx`` (with the folded
    # ``dpre_tail`` rows) is bitwise the det program's.
    "cake_causal_conv1d_train_584216799fed8caece3a_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "kcp_bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "slots": 1,
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``kcp_bwd_head``: this package's copy of the det program
    # ``cake_causal_conv1d_det_train_17cd42fb05079bbbf840_kcp`` (the same build keywords and kernel under this
    # package's prefix; no cross-CTA reduction, so its outputs are bitwise the det program's).
    "cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_head"],
            ["buffer", "head_row_off"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 5,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp",
        "logical": {
            "C": "C",
            "S": "S",
            "cu": "cu",
            "dpre_head": "dpre_head",
            "dy": "dy",
            "head_row_off": "head_row_off",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "num_docs": "num_docs",
            "w": "w",
            "x": "x",
        },
        "role": "kcp_bwd_head",
        "workspace_rule": None,
    },
    # Route ``kcp_bwd_dyn``: the main program of the overlapped backward (no det program: its own kernel module key).
    "cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "dw_acc"],
            ["buffer", "tile_tk"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
            ["buffer", "skip_row_off"],
            ["buffer", "claims"],
            ["buffer", "departs"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "M"],
            ["parameter", "NL"],
            ["parameter", "P"],
            ["parameter", "L"],
            ["parameter", "n_ctas"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 384,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197760,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp",
        "logical": {
            "C": "C",
            "L": "L",
            "M": "M",
            "NL": "NL",
            "P": "P",
            "S": "S",
            "claims": "claims",
            "cu": "cu",
            "departs": "departs",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ctas": "n_ctas",
            "num_docs": "num_docs",
            "skip_row_off": "skip_row_off",
            "tail_row_off": "tail_row_off",
            "tile_tk": "tile_tk",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "role": "kcp_bwd_dyn",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "slots": 4,
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``kcp_bwd_halo``: the tail-row program of the overlapped backward (its own kernel module key).
    "cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 5,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp",
        "logical": {
            "C": "C",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dx": "dx",
            "dy": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "num_docs": "num_docs",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "x": "x",
        },
        "role": "kcp_bwd_halo",
        "workspace_rule": None,
    },
    # Route ``kcp_fwd_halo``: this package's copy of the det program
    # ``cake_causal_conv1d_det_train_138311f76132280b1e94_kcp`` (the same kernel module key under this package's
    # prefix; no cross-CTA reduction, so its outputs are bitwise the det program's): the history-row program of the
    # two-launch forward.  ``y`` of the first ``min(3, len)`` rows of every fragment with a history row
    # (``hist_row_off``, ``-1`` = none), written again from the received ``hist`` rows after a main program that read
    # no history row (the zero-history registry's ``fwd`` route program on the fragments, or the ``kcp_fwd`` program
    # with every ``hist_row_off`` entry ``-1``), with the ``kcp_fwd`` program's instructions in its order, so ``y`` is
    # the one-launch forward's bits.  One CTA per (fragment, block of ``threads x vec`` channels) work item, the head
    # program's rule (:func:`head_grid`).
    "cake_causal_conv1d_train_138311f76132280b1e94_kcp": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 5,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_138311f76132280b1e94_kcp",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_138311f76132280b1e94_kcp",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_138311f76132280b1e94_kcp",
        "logical": {
            "C": "C",
            "cu": "cu",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "num_docs": "num_docs",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "kcp_fwd_halo",
        "workspace_rule": None,
    },
}

# architecture -> KCP kind -> the records the route may launch, in launch
# order.  The forward and the head route carry one program; the backward
# route carries the architecture's fallback program first and the
# big-channel ``_st8`` program second (:func:`selected_modules` launches one);
# the overlapped backward's two routes and the two-launch forward's
# history-row route carry one program each.
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "kcp_bwd": [
            "cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp",
            "cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        ],
        "kcp_bwd_dyn": [
            "cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp",
        ],
        "kcp_bwd_halo": [
            "cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp",
        ],
        "kcp_bwd_head": [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp",
        ],
        "kcp_fwd": [
            "cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp",
        ],
        "kcp_fwd_halo": [
            "cake_causal_conv1d_train_138311f76132280b1e94_kcp",
        ],
    },
    "sm_103a": {
        "kcp_bwd": [
            "cake_causal_conv1d_train_d994e1963ad7724c9c64_kcp",
            "cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        ],
        "kcp_bwd_dyn": [
            "cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp",
        ],
        "kcp_bwd_halo": [
            "cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp",
        ],
        "kcp_bwd_head": [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp",
        ],
        "kcp_fwd": [
            "cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp",
        ],
        "kcp_fwd_halo": [
            "cake_causal_conv1d_train_138311f76132280b1e94_kcp",
        ],
    },
    "sm_107a": {
        "kcp_bwd": [
            "cake_causal_conv1d_train_777e24f8771c3f7abdf8_kcp",
            "cake_causal_conv1d_train_584216799fed8caece3a_kcp",
        ],
        "kcp_bwd_dyn": [
            "cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp",
        ],
        "kcp_bwd_halo": [
            "cake_causal_conv1d_train_ae0c093e1e0ac46aec3f_kcp",
        ],
        "kcp_bwd_head": [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840_kcp",
        ],
        "kcp_fwd": [
            "cake_causal_conv1d_train_675940a7a4fc0dba4b53_kcp",
        ],
        "kcp_fwd_halo": [
            "cake_causal_conv1d_train_138311f76132280b1e94_kcp",
        ],
    },
}

# The KCP kinds (the zero-history kinds are ``cake_jit.KINDS``): the launch
# roles of the det package's KCP registry.
KINDS = ("kcp_fwd", "kcp_bwd", "kcp_bwd_head")
# The overlapped backward's routes: its main program and its tail-row program.
DYN_KIND = "kcp_bwd_dyn"
HALO_KIND = "kcp_bwd_halo"
# The route of the two-launch forward's history-row program: after a main
# program that read no history row, it writes the first rows of every fragment
# with a history row again (module docstring).
FWD_HALO_KIND = "kcp_fwd_halo"
# Every route of ``ROUTES[arch]``.
ROUTE_KINDS = (*KINDS, DYN_KIND, HALO_KIND, FWD_HALO_KIND)
# The history / halo rows every fragment carries (the convolution width minus one).
HALO = WIDTH - 1
# Logical argument names the KCP host flow binds: the zero-history vocabulary
# (``cake_jit.LOGICAL_TENSORS``, spelled out so the registry stays a literal)
# plus the KCP tables.  ``hist``: BF16 history rows ``[n_hist_rows, C]``;
# ``hist_row_off``: int32 ``[n_frag, 3]`` (``-1`` = none); ``dpre_tail``: FP32
# ``[n_tail_rows, C]`` tail derivatives with ``tail_row_off`` int32
# ``[n_frag, 3]``; ``skip_row_off``: int32 ``[n_frag, 3]``, the fragments
# whose last rows the overlapped main program leaves to the tail-row program
# (an entry ``>= 0``); ``dpre_head``: FP32 ``[n_head_rows, C]`` head derivatives
# with ``head_row_off`` int32 ``[n_frag, 3, 3]``; ``acc`` / ``work``: the
# backward's FP32 ``[C, 4]`` accumulators and uint32 ``[n_ch_tiles]`` ticket
# counters (zero before and after every launch); ``tile_tk`` / ``claims`` /
# ``departs``: the overlapped main program's uint32 tile words, claim words
# and departure word (its ``acc`` holds ``slots`` copies of the accumulators;
# :func:`dyn_workspace_layout`).
LOGICAL_TENSORS = (
    "x",
    "w",
    "cu",
    "y",
    "dy",
    "dx",
    "dw",
    "acc",
    "work",
    "hist",
    "hist_row_off",
    "dpre_tail",
    "tail_row_off",
    "skip_row_off",
    "dpre_head",
    "head_row_off",
    "tile_tk",
    "claims",
    "departs",
)
# The scalars: the zero-history scalars (``cake_jit.LOGICAL_SCALARS``) and the
# overlapped main program's work plan (:func:`dyn_plan`).
LOGICAL_SCALARS = (*_zero_history.LOGICAL_SCALARS, "M", "NL", "P", "L", "n_ctas")
# The overlapped main program's channel tile (eight compute warps of 256
# channels), its words per tile (one per warp), its most lanes and channels,
# the smallest unit of tokens its plan considers, a unit's fixed cost in
# tenths of a lane token (start, claim and run switch: about 0.69 us per unit
# against 0.215 us per token of a lane, measured on B200), and the tail's cost
# in tenths of a token per token of a unit: a lane's last unit is the granule
# at which the other CTAs finish a lane whose SM is busy (a held SM or another
# stream's kernel); on B200, 112-token units of 3 x 8192 channels lose 2-6%
# against 64-token units with 16 or 32 SMs held.  ``DYN_IDLE_UNIT``: the
# smaller unit of a plan whose 64-token lanes leave lanes idle, taken when it
# leaves more CTAs without a lane of their own (the CTAs that start by taking
# units from the other lanes absorb the SMs another stream holds): on VR200
# (212 SMs), 3 x 8192 channels of 16384 tokens leave 16 such CTAs with 56-token
# units against 4 with 64-token units, and the backward with 16 / 32 SMs held
# ran 1.130x / 1.055x faster (other phases within 0.6%).
DYN_TILE = 2048
DYN_TILE_WORDS = 8
DYN_MAX_LANES = 256
DYN_MAX_CHANNELS = 32 * DYN_TILE
DYN_UNIT = 64
DYN_IDLE_UNIT = 56
DYN_UNIT_COST_TENTHS = 32
DYN_TAIL_COST_TENTHS = 3


def route_applies(*, device_capability: Sequence[int], kind: str) -> bool:
    """Whether the verified export owns one KCP training call on a device."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return kind in ROUTES.get(arch, {})


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one KCP route on ``arch`` (every program the route may launch).

    The forward and the head route launch their one program; the backward
    launches the one :func:`selected_modules` picks for the call's channel
    count.
    """

    if kind not in ROUTE_KINDS:
        raise ValueError(f"kind must be one of {ROUTE_KINDS}, got {kind!r}")
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d KCP training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record, ``None`` when the record carries none.

    A backward record without a rule (no ``select_rule`` key, or ``None``) is
    its route's fallback program.  ``{"kind": "channels_at_least", "channels":
    {arch: C_min, ...}}`` owns the calls whose channel count is at least
    ``channels[arch]`` on the architectures it names and never hits on the
    others.  Forward and head records carry no rule.
    """

    return record.get("select_rule")


def selected_modules(
    arch: str, kind: str, *, C: int, S: int | None = None
) -> tuple[str, ...]:
    """Ordered module names one KCP call launches on ``arch`` for ``C`` channels.

    The forward and the head route launch every program of their route
    (:func:`route_modules`, one each).  The backward route carries the
    architecture's fallback program and the big-channel program and launches
    exactly one: the record whose :func:`select_rule` admits ``C`` on ``arch``
    when one does, else the route's single fallback record (the one without a
    rule) -- the rule of the zero-history backward.  The choice is a pure
    function of the architecture and the channel count (``S`` is accepted and
    unused, and the host never reads the data), so a given ``(S, C, device)``
    always runs the same program (``dx`` is that program's bits; the cross-CTA
    sum of ``dweight`` is unordered).
    Several admitting records, or zero or several fallbacks, are registry
    errors (``RuntimeError``); an unknown rule kind is a ``ValueError``.
    """

    names = route_modules(arch, kind)
    if kind != "kcp_bwd":
        return names
    positive: list[str] = []
    fallback: list[str] = []
    for name in names:
        rule = select_rule(MODULES[name])
        if rule is None:
            fallback.append(name)
        elif _zero_history._admits(rule, arch=arch, C=C):
            positive.append(name)
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported KCP backward route on {arch} declares {len(fallback)} fallback "
            "programs (records without a select_rule); expected exactly one"
        )
    if len(positive) > 1:
        raise RuntimeError(
            f"the exported KCP backward route on {arch} selects {len(positive)} programs "
            f"for C={int(C)} ({positive}); expected at most one"
        )
    if positive:
        return (positive[0],)
    return (fallback[0],)


def workspace_owners(arch: str) -> tuple[dict[str, Any], ...]:
    """The KCP backward module records of ``arch`` (each owns an ``accumulators`` workspace), in route order."""

    owners = tuple(MODULES[name] for name in route_modules(arch, "kcp_bwd"))
    if not owners:
        raise RuntimeError(
            f"the exported KCP backward route on {arch} declares no workspace owner; "
            "expected at least one"
        )
    for owner in owners:
        rule = owner["workspace_rule"]
        if rule is None or rule["kind"] != "accumulators":
            raise RuntimeError(
                f"the KCP backward program {owner['kernel']} on {arch} declares the workspace rule "
                f"{rule!r}; every KCP backward program of this package owns an 'accumulators' workspace"
            )
    return owners


def workspace_record(arch: str) -> dict[str, Any]:
    """The capacity owner of the KCP backward accumulator workspace on ``arch``.

    As for the zero-history route: the owners share one build geometry (one
    legacy ``n_ch_tiles`` per ``C``), so their workspaces differ only in the
    accumulator bytes per channel; the first record with the most bounds them
    all (every KCP program accumulates ``[C, 4]`` FP32 words).
    """

    owners = workspace_owners(arch)
    geometries = {
        tuple(sorted(owner["grid_rule"]["geometry"].items())) for owner in owners
    }
    if len(geometries) != 1:
        raise RuntimeError(
            f"the KCP backward programs on {arch} declare {len(geometries)} build "
            "geometries; one workspace serves them only when they share one"
        )

    def per_channel(owner: dict[str, Any]) -> int:
        rule = owner["workspace_rule"]
        return (
            _zero_history.accumulator_words_per_channel(owner)
            * _zero_history.ACCUMULATOR_DTYPES[rule["acc"]]
        )

    best = owners[0]
    for owner in owners[1:]:
        if per_channel(owner) > per_channel(best):
            best = owner
    return best


def workspace_bytes(arch: str, C: int) -> int:
    """Bytes of the KCP backward's ``acc`` / ``work`` workspace for ``C`` channels on ``arch``: the largest
    :func:`accumulator_layout` over the route's programs under the legacy channel tile rule the KCP path plans
    with (``ct_vec = min(C / 8, 128)``)."""

    nbytes = 0
    for owner in workspace_owners(arch):
        _, _, n_ch_tiles = channel_split(
            C, geometry=owner["grid_rule"]["geometry"], legacy=True
        )
        nbytes = max(
            nbytes, int(accumulator_layout(owner, C, n_ch_tiles)["total_bytes"])
        )
    return nbytes


def dyn_applies(C: int, S: int) -> bool:
    """Whether the overlapped backward runs a call of ``C`` channels and ``S`` tokens (at least one token and at most
    :data:`DYN_MAX_CHANNELS` channels); the ``kcp_bwd`` route runs the others."""

    return int(S) >= 1 and int(C) <= DYN_MAX_CHANNELS


def dyn_plan(S: int, C: int, sms: int) -> dict[str, int]:
    """The overlapped main program's work plan on a device of ``sms`` SMs: ``P`` lanes per tile of
    :data:`DYN_TILE` channels (as many as give every SM's CTA a lane, at most one per unit of tokens and
    :data:`DYN_MAX_LANES` lanes in all), each of ``NL`` units of ``M`` tokens, ``L`` lanes, ``n_ctas = max(L, sms)``
    CTAs (the grid) and ``n_tiles``.  Lane ``j`` of a tile starts at token ``j * NL * M``, and a CTA whose lane is
    done takes units from the ends of the other lanes.  When the plan of :data:`DYN_UNIT` tokens leaves the last
    lane of each tile without tokens (``ceil(S / (NL * M)) < P``), that plan is taken, or the plan of
    :data:`DYN_IDLE_UNIT` tokens when it leaves more idle CTAs (``n_ctas`` less the lanes with tokens; the
    :data:`DYN_UNIT` plan on a tie): the empty lanes' CTAs and the spare CTAs take units from the start, and the
    small units balance the lanes better than larger units shorten them.  Otherwise every lane is busy, the lane
    bound sets the time, and ``M`` is the multiple of 8 from
    :data:`DYN_UNIT` to twice it whose plan fits the lanes' 16-bit claim counts at the least lane cost ``NL * M``
    tokens plus ``NL`` units of :data:`DYN_UNIT_COST_TENTHS` plus the tail's ``M`` tokens of
    :data:`DYN_TAIL_COST_TENTHS` (the smaller ``M`` on a tie); when none fits, ``M`` doubles from
    ``4 * DYN_UNIT`` until a plan fits.  A pure function of ``(S, C, sms)``; ``dx`` does not depend on it."""

    S, C, sms = int(S), int(C), int(sms)
    if S < 1 or C < 1 or C > DYN_MAX_CHANNELS:
        raise ValueError(
            f"the overlapped backward plans 1 <= C <= {DYN_MAX_CHANNELS} and S >= 1, got C={C}, S={S}"
        )
    n_tiles = -(-C // DYN_TILE)

    def plan_for(M: int) -> dict[str, int] | None:
        P = max(1, min(sms // n_tiles, -(-S // M), DYN_MAX_LANES // n_tiles))
        per_lane = -(-S // P)
        NL = -(-per_lane // M)
        L = n_tiles * P
        n_ctas = max(L, sms)
        if (L + 1) * NL + n_ctas > 65535 or NL > 32768:
            return None
        return {"M": M, "NL": NL, "P": P, "L": L, "n_ctas": n_ctas, "n_tiles": n_tiles}

    def idle_ctas(p: dict[str, int]) -> int:
        """The plan's CTAs without a lane of their own that has tokens."""
        return p["n_ctas"] - p["n_tiles"] * min(p["P"], -(-S // (p["NL"] * p["M"])))

    base = plan_for(DYN_UNIT)
    if base is not None and -(-S // (base["NL"] * DYN_UNIT)) < base["P"]:
        # measured on GB300, GB200 and VR200: with idle lanes, 64-token units beat the larger units the cost picks;
        # 56-token units that leave more CTAs idle absorb held SMs better (VR200, DYN_IDLE_UNIT)
        small = plan_for(DYN_IDLE_UNIT)
        if small is not None and idle_ctas(small) > idle_ctas(base):
            return small
        return base
    fits = [
        p for p in map(plan_for, range(DYN_UNIT, 2 * DYN_UNIT + 1, 8)) if p is not None
    ]
    if fits:
        return min(
            fits,
            key=lambda p: (
                p["NL"] * (10 * p["M"] + DYN_UNIT_COST_TENTHS)
                + DYN_TAIL_COST_TENTHS * p["M"],
                p["M"],
            ),
        )
    M = 4 * DYN_UNIT
    p = plan_for(M)
    while p is None:
        M *= 2
        p = plan_for(M)
    return p


def dyn_workspace_layout(
    record: dict[str, Any], C: int, n_tiles: int, L: int
) -> dict[str, Any]:
    """:func:`accumulator_layout` of the overlapped main program: ``slots`` copies of the ``[C, 4]`` FP32
    accumulators, then ``DYN_TILE_WORDS x n_tiles`` tile words, ``L`` claim words and one departure word
    (``tile_tk`` is ``work[:DYN_TILE_WORDS * n_tiles]``, then ``claims``, then ``departs``)."""

    return accumulator_layout(record, C, DYN_TILE_WORDS * int(n_tiles) + int(L) + 1)


def head_grid(record: dict[str, Any], *, n_frag: int, C: int) -> tuple[int, int, int]:
    """The grid of the head program, of the tail-row program and of the history-row program: ``max(1, n_frag x ceil(C
    / (threads x vec)))`` work items over the record's geometry."""

    geometry = record["grid_rule"]["geometry"]
    channels_per_cta = int(geometry["threads"]) * int(geometry["vec"])
    if channels_per_cta <= 0:
        raise ValueError(
            "the head record's geometry must cover at least one channel per CTA"
        )
    items = int(n_frag) * (-(-int(C) // channels_per_cta))
    return (max(1, items), 1, 1)


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    return _zero_history.device_arch(device_index)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of KCP module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake causal conv1d KCP training module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of KCP record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of the package's ``build.json`` without a prebuilt file
    (the zero-history and the KCP entries alike; for ahead-of-time warm-up through FlashInfer's
    ``build_jit_specs``)."""

    return cake_build.jit_specs(arch)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    device_index: int,
    grid: tuple[int, int, int] | None = None,
) -> None:
    """Launch one exported KCP module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is the
    module's :func:`launch_grid` on ``device_index`` (``ctas_per_sm x SM
    count``) unless ``grid`` gives it (the head program's work-item grid,
    :func:`head_grid`), and the kernel runs on the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    if grid is None:
        grid = launch_grid(record, device_index=device_index)
    grid_values = {
        "grid_x": int(grid[0]),
        "grid_y": int(grid[1]),
        "grid_z": int(grid[2]),
    }
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
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
    "CHANNEL_MULTIPLE",
    "DYN_KIND",
    "DYN_MAX_CHANNELS",
    "DYN_MAX_LANES",
    "DYN_TILE",
    "DYN_TILE_WORDS",
    "DYN_IDLE_UNIT",
    "DYN_UNIT",
    "DYN_TAIL_COST_TENTHS",
    "DYN_UNIT_COST_TENTHS",
    "FWD_HALO_KIND",
    "HALO",
    "HALO_KIND",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "ROUTES",
    "ROUTE_KINDS",
    "SELECT_KINDS",
    "WIDTH",
    "accumulator_layout",
    "arch_for_capability",
    "channel_split",
    "device_arch",
    "dyn_applies",
    "dyn_plan",
    "dyn_workspace_layout",
    "head_grid",
    "launch_grid",
    "library_name",
    "load",
    "plan",
    "route_applies",
    "route_modules",
    "run_module",
    "select_rule",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "workspace_bytes",
    "workspace_owners",
    "workspace_record",
]
