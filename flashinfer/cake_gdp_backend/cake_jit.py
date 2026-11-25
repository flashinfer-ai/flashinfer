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

Host plan and loader glue of the Cake Gated DeltaProduct (GDP) kernels
(``K = 128``, ``V = 64``, ``n <= 4`` Householder updates per token, FP32
recurrence): the decode / MTP-verification programs (``role`` ``"decode"``;
SM90 / SM100 / SM103 / SM107) and the chunked varlen prefill programs (``role``
``"prefill"``; SM90 / SM100 / SM103).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, the build
keywords it was rendered with, its launch record, grid rule, selection rule and
the kernel-parameter -> logical-argument map.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's
JIT from the same sources and flags, verifying every source digest before the
first build.

A route is the list of module records one public call may launch.  Decode:
``decode_nh<n>`` for ``n`` Householder updates per token with the q / k L2
normalization fused in the kernel.  One record is launched per call: the
static-token-count program of the call's ``T`` whose band admits the call's
``B x HV`` state heads on the device's SM count (``selected_module``), else the
runtime-token-count program of the route (``T`` up to its ``max_t_steps``).
Batch size and head count are runtime scalars, so a changed problem size never
recompiles anything.  Two kernel bodies ship: the lane-owns-rows body
(``variant`` ``l8r4`` / ``l8r4p`` / ``l8r4w5`` / ``l8r2`` / ``l4r2``, one
CTA per batch row and head, and SM90's FP16 ``l8r2h``, two)
and, for ``T = 1`` with ``B x HV <= 160``, the row-partition body
(``splitv8_tile8``, eight CTAs per state head); each record names its body
(``body``: ``lanerow`` for the lane-owns-rows body, ``splitv`` for the row
partition).  A band may differ per architecture (SM90 has neither the
one-wave ``l8r4p`` program nor the one-wave-of-5 ``l8r4w5`` program at
``T = 1``, owns the one-CTA-per-SM
rows of ``T = 4`` with ``l4r2`` and runs ``l8r2`` at ``T = 16`` up to one CTA
per SM instead of two): a selection cell then names the ``arches``
it applies to, and a
cell without ``arches`` applies on every architecture its record is delivered
for.

A decode record reads and writes a state pool of one dtype, its build's
``state_dtype``: ``"f16"`` for an FP16 pool, absent for an FP32 pool (the
default).  Every FP16 record but the row-split programs below is the twin of
the FP32 record of the same route, body, variant and token count, with the same
band less the cells of those programs, and is delivered on the architectures of
that record (SM90 has the FP16 twins of its own ``l4r2`` program at ``T = 4``
and of its own build of the row-partition program, and its own FP16 build of
the ``l8r4`` program of 6 tokens).  Its
stores round the FP32 state with the hardware ``cvt.rs`` instruction on SM100 /
SM103 / SM107 and with the same rounding in integer arithmetic on SM90, which
has no ``cvt.rs``.  Its kernel takes two more arguments, the int64 ``rand_seed``
tensor it reads on the device and the Philox round count ``philox_rounds``
(``0`` rounds to nearest even and reads no seed).  The dispatch functions
take the pool's dtype as ``state_dtype`` (``torch.float32`` by default) and
consider the records of that dtype only, so an FP16 pool runs the twin of the
program an FP32 pool would run, or a row-split program.

An FP16 call also has a launch class: its round count (``philox_rounds``, 0
without a seed) and whether it stores per-token snapshots
(``ssm_state_indices`` given).  A selection cell of an FP16 record may bound
the round count (``rounds_at_least`` / ``rounds_at_most``, inclusive) and name
the snapshot use (``snapshot``); an absent key admits every value.  The
``l8r4`` program of 16 tokens and the runtime program of three Householder
updates have a sibling built with ``snapshot_noise``, and on SM90 so does the
``l8r4`` program of 6 tokens.  The sibling
draws a launch's random bits once into shared memory instead of at each store,
which pays only when the launch stores snapshots; it keeps its program's body,
variant and band and stores the same bits.  Its cells admit the stochastic
calls with snapshots of the measured cell, and its plain program's cells admit
every other call of the band, so every call still selects exactly one program
and an FP16 pool still runs the body and variant an FP32 pool would run.
The FP16 row-partition program of SM100 / SM103 / SM107 has a sibling built
with ``draw_share`` (SM90 ships none): each
Philox draw of its store is computed once and shared by a lane shuffle, so it
stores the same bits.  Its cell admits the stochastic calls of the band
(``rounds_at_least``), and its plain program keeps the calls that round to
nearest even (``rounds_at_most``).

SM90 also ships FP16-only row-split programs (``variant`` ``l8r2h``: the
``l8r2`` lane layout with each state head cut over two CTAs of 128 threads,
grid rule ``ctas_per_state`` 2), one for each static token count of 2 to 8
and 16 and one runtime program for each ``n``; no route ships their FP32
build.  Their
cells admit the small-grid calls on which they were measured faster, with
wave bounds that are multiples of 1/2 (``k x SM`` state heads run on
``2 k`` CTAs per SM).  Such a call runs the row-split program of its token
count (static or runtime) instead of the program or sibling it would run
otherwise, whose cells admit every other call of the band, so every call
still selects exactly one program.

Prefill: the ``prefill`` route holds the single-pass programs of two kernel
bodies: the V6 fork-order body (``body`` ``prefill_sm90_v6``: the plain, the
initial-state and the pool-indexed flag sets, final state stored, and their
``split_items`` twins, which the SM90 V6 WARM-UP SPLIT PLAN below feeds) and the
merged chunked prefill body (``body`` ``prefill_sm90``: the unit programs and
the ``split_items`` programs); the BODY ROUTE OF RECORD below
(``prefill_body_class``) picks the body of every sequence from its chunk
count, ``H``, ``n`` and the SM count, and a call launches once per body
present (``prefill_launch_plan``).  Within a body, one program per flag set
the public call may need (``select_rule`` ``prefill_flags``: initial state none / given /
pool-indexed, final state stored or not, ``g=None`` / ``beta=None`` (the
``unit_gates`` / ``unit_beta`` programs, from zeros; both absent is no shipped
program); ``selected_prefill_module``), each walking every (sequence, head)
tile with ``min(SM count, N x H)`` persistent CTAs (grid rule ``tiles``); the
route also holds that body's ``split_items`` programs (the gate-saturation
split: a long sequence runs as ``p`` independent items per head over the
host's item table -- formal 9, ``output_slot_indices``; ``total_tiles`` = items
-- and ``selected_prefill_module(..., split_items=True)`` names them; the
facade takes them for the merged-class sequences of a call, which the split
rules of record below always cut).  Those are the SM90 programs.  On SM100 /
SM103 the route holds the programs of the Blackwell body (``body``
``prefill_bw_v6``): one launch serves every flag set (a ``flags`` word reads the
initial state and stores the final state) over a host item table whose order
places the work on the persistent CTAs; the BLACKWELL V6 LAUNCH PLAN below
(``prefill_v6_plan``) builds the table and picks the build, which the
``v6_launch`` select rule names (``selected_v6_module``).  The Blackwell
body's programs come in state forms, each with its block and split build: the
dense FP32 programs (one ``[N, H, 64, 128]`` record per sequence) and the pool
programs of the build keys ``use_state_indices`` / ``use_output_state_indices``
(``state_indices`` / ``output_state_indices`` slot tables over
``[pool, H, 64, 128]`` pools whose slot strides are kernel arguments; a read
slot outside the initial pool starts from zero, a write slot outside the output
pool stores nothing), for an FP32, a BF16 and an FP16 pool (``state_dtype``
``"bf16"`` / ``"f16"``: the state is read widened exactly and its final value
rounded once at the store, to nearest even or, with ``rand_seed`` and
``philox_rounds``, stochastically by the hardware ``cvt.rs``).  A 16-bit record
is the twin of the FP32 pool record of the same split; the facade runs a dense
16-bit call on the pool program with identity slots.  ``selected_v6_module``
takes the state form (``state_dtype``, ``pool``) beside the plan.
Every grid is computed from host integers; no tensor is read on the host.

The program name ``cake_gdp_<20 hex>`` is the first twenty hex digits of the
SHA-256 of the compact JSON ``{"build": <build keywords, sorted>, "module":
<dotted kernel module>}`` of the kernel the program was rendered from; the
record's ``build`` spells those keywords: ``variant`` (the body's geometry:
``l8r4`` = eight lanes per state row, four rows per lane; ``l8r4p`` and
``l8r4w5`` = that geometry with the producer preload, for one wave of four
CTAs per SM and for the many-wave rows (``B x HV > 64 x SM
count``); ``l8r2``; ``l8r2h`` = ``l8r2`` with a head cut over two CTAs;
``l4r2``; ``l4r4`` for the runtime programs; ``splitv8_tile8`` = eight CTAs of
eight rows
per head), ``num_householder`` (``n``),
``t_steps`` (a fixed token count compiled in; ``None`` for a runtime token
count up to ``max_t_steps``), ``use_qk_l2norm``, on an FP16-pool record
``state_dtype``, on a sibling ``snapshot_noise`` or ``draw_share``, and
``arch`` on an SM90 build whose program differs from that of the other
architectures (the row-partition programs and the FP16 ``l8r4`` program of 6
tokens).  Prefill: the six flags
``use_initial_state``, ``use_state_indices``, ``store_final_state``,
``unit_gates``, ``unit_beta`` and ``segment_pass`` beside the defaults of the
kernel's build signature (the I/O dtype ``bf16``, the WY solve arm, the O
store, ...; the split programs spell ``split_items`` too).
"""

from __future__ import annotations

import array
import bisect
import collections
import ctypes
import functools
import hashlib
import heapq
import itertools
import operator
import struct
import warnings
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import torch
import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_gdp_068a895d0c8fa77ed2f9": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "splitv",
        "build": {
            "arch": "sm_90a",
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "splitv8_tile8",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 8,
            "kind": "state_heads",
            "tile_v": 8,
        },
        "kernel": "cake_gdp_068a895d0c8fa77ed2f9",
        "kernel_symbol": "kernel_cake_gdp_068a895d0c8fa77ed2f9",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 6400,
            "use_pdl": False,
        },
        "library": "cake_gdp_068a895d0c8fa77ed2f9",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": 160,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_8f8062d89b8b871f1e2f": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "splitv",
        "build": {
            "arch": "sm_90a",
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "splitv8_tile8",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 8,
            "kind": "state_heads",
            "tile_v": 8,
        },
        "kernel": "cake_gdp_8f8062d89b8b871f1e2f",
        "kernel_symbol": "kernel_cake_gdp_8f8062d89b8b871f1e2f",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2304,
            "use_pdl": False,
        },
        "library": "cake_gdp_8f8062d89b8b871f1e2f",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": 160,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_0c31b01fa2a76d426c5f": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_0c31b01fa2a76d426c5f",
        "kernel_symbol": "kernel_cake_gdp_0c31b01fa2a76d426c5f",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33536,
            "use_pdl": False,
        },
        "library": "cake_gdp_0c31b01fa2a76d426c5f",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_0f9d4678110f025383b2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "l8r4w5",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_0f9d4678110f025383b2",
        "kernel_symbol": "kernel_cake_gdp_0f9d4678110f025383b2",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2944,
            "use_pdl": False,
        },
        "library": "cake_gdp_0f9d4678110f025383b2",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 160,
            "at_most": None,
            "ctas_per_sm_above": 64,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_159ecba6294f1c73ed91": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 3,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_159ecba6294f1c73ed91",
        "kernel_symbol": "kernel_cake_gdp_159ecba6294f1c73ed91",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": False,
        },
        "library": "cake_gdp_159ecba6294f1c73ed91",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_16d24c9e19fa73fe57ee": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 16,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_16d24c9e19fa73fe57ee",
        "kernel_symbol": "kernel_cake_gdp_16d24c9e19fa73fe57ee",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 45440,
            "use_pdl": False,
        },
        "library": "cake_gdp_16d24c9e19fa73fe57ee",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
                "rounds_at_most": 4,
            },
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
                "snapshot": False,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "kind": "state_heads_between",
                "rounds_at_most": 4,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
                "snapshot": False,
            },
        ],
    },
    "cake_gdp_1a3d01a5826508d64a06": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 7,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_1a3d01a5826508d64a06",
        "kernel_symbol": "kernel_cake_gdp_1a3d01a5826508d64a06",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 19968,
            "use_pdl": False,
        },
        "library": "cake_gdp_1a3d01a5826508d64a06",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_245414df8449fb1825a1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90",
        "build": {
            "debug_trace": False,
            "desc_iter": False,
            "exact_diag": True,
            "inverse": "blk32",
            "io_dtype": "bf16",
            "l_tile": None,
            "mask_fast": False,
            "o_store": "st_global",
            "segment_pass": False,
            "solve_ablate": False,
            "solve_block": 4,
            "solve_ilp": 1,
            "solve_overlap": True,
            "solve_prefetch": 0,
            "split_decay": True,
            "split_items": True,
            "split_state": None,
            "split_w": None,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": False,
            "wait_hint": None,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_245414df8449fb1825a1",
        "kernel_symbol": "kernel_cake_gdp_245414df8449fb1825a1",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226432,
            "use_pdl": False,
        },
        "library": "cake_gdp_245414df8449fb1825a1",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": False,
        },
    },
    "cake_gdp_27eb88658ee4bdc4ae35": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 4,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_27eb88658ee4bdc4ae35",
        "kernel_symbol": "kernel_cake_gdp_27eb88658ee4bdc4ae35",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 11392,
            "use_pdl": False,
        },
        "library": "cake_gdp_27eb88658ee4bdc4ae35",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": [
                    "sm_100a",
                    "sm_103a",
                    "sm_107a",
                ],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": [
                    "sm_90a",
                ],
                "at_most": None,
                "ctas_per_sm_above": 1,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_3220d37914678385a837": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90",
        "build": {
            "debug_trace": False,
            "desc_iter": False,
            "exact_diag": True,
            "inverse": "blk32",
            "io_dtype": "bf16",
            "l_tile": None,
            "mask_fast": False,
            "o_store": "st_global",
            "segment_pass": False,
            "solve_ablate": False,
            "solve_block": 4,
            "solve_ilp": 1,
            "solve_overlap": True,
            "solve_prefetch": 0,
            "split_decay": True,
            "split_items": True,
            "split_state": None,
            "split_w": None,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
            "wait_hint": None,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_3220d37914678385a837",
        "kernel_symbol": "kernel_cake_gdp_3220d37914678385a837",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226432,
            "use_pdl": False,
        },
        "library": "cake_gdp_3220d37914678385a837",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
        },
    },
    "cake_gdp_32a3f37f5ef3015142a5": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 5,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_32a3f37f5ef3015142a5",
        "kernel_symbol": "kernel_cake_gdp_32a3f37f5ef3015142a5",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 14336,
            "use_pdl": False,
        },
        "library": "cake_gdp_32a3f37f5ef3015142a5",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_3342c82179fa758b0570": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 2,
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_3342c82179fa758b0570",
        "kernel_symbol": "kernel_cake_gdp_3342c82179fa758b0570",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33152,
            "use_pdl": False,
        },
        "library": "cake_gdp_3342c82179fa758b0570",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": None,
    },
    "cake_gdp_3425d6511e4eab4896cd": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 4,
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_3425d6511e4eab4896cd",
        "kernel_symbol": "kernel_cake_gdp_3425d6511e4eab4896cd",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 57856,
            "use_pdl": False,
        },
        "library": "cake_gdp_3425d6511e4eab4896cd",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": None,
    },
    "cake_gdp_3cbdef34ce5b580c2216": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 8,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_3cbdef34ce5b580c2216",
        "kernel_symbol": "kernel_cake_gdp_3cbdef34ce5b580c2216",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 22784,
            "use_pdl": False,
        },
        "library": "cake_gdp_3cbdef34ce5b580c2216",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_450bb0db3b79acafa419": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 16,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_450bb0db3b79acafa419",
        "kernel_symbol": "kernel_cake_gdp_450bb0db3b79acafa419",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 45440,
            "use_pdl": False,
        },
        "library": "cake_gdp_450bb0db3b79acafa419",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": [
                    "sm_100a",
                    "sm_103a",
                    "sm_107a",
                ],
                "at_most": None,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": [
                    "sm_90a",
                ],
                "at_most": None,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_46ef1a68db53b4c532e9": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 5,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_46ef1a68db53b4c532e9",
        "kernel_symbol": "kernel_cake_gdp_46ef1a68db53b4c532e9",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 14336,
            "use_pdl": False,
        },
        "library": "cake_gdp_46ef1a68db53b4c532e9",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_4d40c98af90453032200": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "l8r4w5",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_4d40c98af90453032200",
        "kernel_symbol": "kernel_cake_gdp_4d40c98af90453032200",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2944,
            "use_pdl": False,
        },
        "library": "cake_gdp_4d40c98af90453032200",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 160,
            "at_most": None,
            "ctas_per_sm_above": 64,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_585a5ea4cbdd07f1b2a8": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 3,
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_585a5ea4cbdd07f1b2a8",
        "kernel_symbol": "kernel_cake_gdp_585a5ea4cbdd07f1b2a8",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 45440,
            "use_pdl": False,
        },
        "library": "cake_gdp_585a5ea4cbdd07f1b2a8",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": None,
    },
    "cake_gdp_5970bfa6676087a3ce92": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 8,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_5970bfa6676087a3ce92",
        "kernel_symbol": "kernel_cake_gdp_5970bfa6676087a3ce92",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 22784,
            "use_pdl": False,
        },
        "library": "cake_gdp_5970bfa6676087a3ce92",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_5bc4ef21b6e7aefed05f": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 4,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_5bc4ef21b6e7aefed05f",
        "kernel_symbol": "kernel_cake_gdp_5bc4ef21b6e7aefed05f",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 27776,
            "use_pdl": False,
        },
        "library": "cake_gdp_5bc4ef21b6e7aefed05f",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_6bb51453cd15a43da9c2": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 3,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_6bb51453cd15a43da9c2",
        "kernel_symbol": "kernel_cake_gdp_6bb51453cd15a43da9c2",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 24960,
            "use_pdl": False,
        },
        "library": "cake_gdp_6bb51453cd15a43da9c2",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_717ce3c684b41c37e673": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 4,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_717ce3c684b41c37e673",
        "kernel_symbol": "kernel_cake_gdp_717ce3c684b41c37e673",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 74240,
            "use_pdl": False,
        },
        "library": "cake_gdp_717ce3c684b41c37e673",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_72b6872c98ed9abf0bdf": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 7,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_72b6872c98ed9abf0bdf",
        "kernel_symbol": "kernel_cake_gdp_72b6872c98ed9abf0bdf",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 36352,
            "use_pdl": False,
        },
        "library": "cake_gdp_72b6872c98ed9abf0bdf",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 0.5,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_78fc9caafee16e2af047": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 1,
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_78fc9caafee16e2af047",
        "kernel_symbol": "kernel_cake_gdp_78fc9caafee16e2af047",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_gdp_78fc9caafee16e2af047",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": None,
    },
    "cake_gdp_7e4385817ba59915bc61": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 8,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_7e4385817ba59915bc61",
        "kernel_symbol": "kernel_cake_gdp_7e4385817ba59915bc61",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 39168,
            "use_pdl": False,
        },
        "library": "cake_gdp_7e4385817ba59915bc61",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 0.5,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_8e26e2c2d0cb0a7fbee9": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 16,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_8e26e2c2d0cb0a7fbee9",
        "kernel_symbol": "kernel_cake_gdp_8e26e2c2d0cb0a7fbee9",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 61824,
            "use_pdl": False,
        },
        "library": "cake_gdp_8e26e2c2d0cb0a7fbee9",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 0.5,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_8f4ad44e524af3f3cbb9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "l8r4p",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_8f4ad44e524af3f3cbb9",
        "kernel_symbol": "kernel_cake_gdp_8f4ad44e524af3f3cbb9",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2944,
            "use_pdl": False,
        },
        "library": "cake_gdp_8f4ad44e524af3f3cbb9",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 160,
            "at_most": None,
            "ctas_per_sm_at_most": 4,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_9bfd49d6bbbb3d849f30": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_9bfd49d6bbbb3d849f30",
        "kernel_symbol": "kernel_cake_gdp_9bfd49d6bbbb3d849f30",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33536,
            "use_pdl": False,
        },
        "library": "cake_gdp_9bfd49d6bbbb3d849f30",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 0.5,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_a226c899893599e679a3": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_a226c899893599e679a3",
        "kernel_symbol": "kernel_cake_gdp_a226c899893599e679a3",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 17152,
            "use_pdl": False,
        },
        "library": "cake_gdp_a226c899893599e679a3",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_11085a9519b214993653": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "arch": "sm_90a",
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_11085a9519b214993653",
        "kernel_symbol": "kernel_cake_gdp_11085a9519b214993653",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 17152,
            "use_pdl": False,
        },
        "library": "cake_gdp_11085a9519b214993653",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
                "rounds_at_most": 4,
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
                "snapshot": False,
            },
        ],
    },
    "cake_gdp_8aabe827188edf904960": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "snapshot_noise": True,
            "state_dtype": "f16",
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_8aabe827188edf904960",
        "kernel_symbol": "kernel_cake_gdp_8aabe827188edf904960",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 33536,
            "use_pdl": False,
        },
        "library": "cake_gdp_8aabe827188edf904960",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
            "rounds_at_least": 5,
            "snapshot": True,
        },
    },
    "cake_gdp_b047f2f1a20630e6f36b": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "snapshot_noise": True,
            "state_dtype": "f16",
            "t_steps": 16,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_b047f2f1a20630e6f36b",
        "kernel_symbol": "kernel_cake_gdp_b047f2f1a20630e6f36b",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 61824,
            "use_pdl": False,
        },
        "library": "cake_gdp_b047f2f1a20630e6f36b",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
                "snapshot": True,
            },
        ],
    },
    "cake_gdp_b81854850bf4b769295c": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90",
        "build": {
            "debug_trace": False,
            "desc_iter": False,
            "exact_diag": True,
            "inverse": "blk32",
            "io_dtype": "bf16",
            "l_tile": None,
            "mask_fast": False,
            "o_store": "st_global",
            "segment_pass": False,
            "solve_ablate": False,
            "solve_block": 4,
            "solve_ilp": 1,
            "solve_overlap": True,
            "solve_prefetch": 0,
            "split_decay": True,
            "split_items": True,
            "split_state": None,
            "split_w": None,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": True,
            "wait_hint": None,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_b81854850bf4b769295c",
        "kernel_symbol": "kernel_cake_gdp_b81854850bf4b769295c",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226432,
            "use_pdl": False,
        },
        "library": "cake_gdp_b81854850bf4b769295c",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_aa7f7810c60fe76c14a6": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 2,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_aa7f7810c60fe76c14a6",
        "kernel_symbol": "kernel_cake_gdp_aa7f7810c60fe76c14a6",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 49536,
            "use_pdl": False,
        },
        "library": "cake_gdp_aa7f7810c60fe76c14a6",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_baca2c2d21cf59fc2d3c": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 4,
            "use_qk_l2norm": True,
            "variant": "l4r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_baca2c2d21cf59fc2d3c",
        "kernel_symbol": "kernel_cake_gdp_baca2c2d21cf59fc2d3c",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 11392,
            "use_pdl": False,
        },
        "library": "cake_gdp_baca2c2d21cf59fc2d3c",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 1,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_b245506c43aed8c3e6b4": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 4,
            "use_qk_l2norm": True,
            "variant": "l4r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_b245506c43aed8c3e6b4",
        "kernel_symbol": "kernel_cake_gdp_b245506c43aed8c3e6b4",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 11392,
            "use_pdl": False,
        },
        "library": "cake_gdp_b245506c43aed8c3e6b4",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 0.5,
            "ctas_per_sm_at_most": 1,
            "kind": "state_heads_between",
            "rounds_at_most": 4,
        },
    },
    "cake_gdp_bb55d7d5e4db0d95acce": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 8,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_bb55d7d5e4db0d95acce",
        "kernel_symbol": "kernel_cake_gdp_bb55d7d5e4db0d95acce",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 39168,
            "use_pdl": False,
        },
        "library": "cake_gdp_bb55d7d5e4db0d95acce",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_bc76e12e0c247b44a27d": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 7,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_bc76e12e0c247b44a27d",
        "kernel_symbol": "kernel_cake_gdp_bc76e12e0c247b44a27d",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 36352,
            "use_pdl": False,
        },
        "library": "cake_gdp_bc76e12e0c247b44a27d",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_c3cf32174475946d9c35": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 2,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_c3cf32174475946d9c35",
        "kernel_symbol": "kernel_cake_gdp_c3cf32174475946d9c35",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 22144,
            "use_pdl": False,
        },
        "library": "cake_gdp_c3cf32174475946d9c35",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_c51f69d9ac126bbaab36": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 16,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_c51f69d9ac126bbaab36",
        "kernel_symbol": "kernel_cake_gdp_c51f69d9ac126bbaab36",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 45440,
            "use_pdl": False,
        },
        "library": "cake_gdp_c51f69d9ac126bbaab36",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": [
                    "sm_100a",
                    "sm_103a",
                    "sm_107a",
                ],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": [
                    "sm_90a",
                ],
                "at_most": None,
                "ctas_per_sm_above": 1,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_c8580552e73f2264f8eb": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90",
        "build": {
            "debug_trace": False,
            "desc_iter": False,
            "exact_diag": True,
            "inverse": "blk32",
            "io_dtype": "bf16",
            "l_tile": None,
            "mask_fast": False,
            "o_store": "st_global",
            "segment_pass": False,
            "solve_ablate": False,
            "solve_block": 4,
            "solve_ilp": 1,
            "solve_overlap": True,
            "solve_prefetch": 0,
            "split_decay": True,
            "split_items": False,
            "split_state": None,
            "split_w": None,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": True,
            "use_initial_state": False,
            "use_state_indices": False,
            "wait_hint": None,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_c8580552e73f2264f8eb",
        "kernel_symbol": "kernel_cake_gdp_c8580552e73f2264f8eb",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226304,
            "use_pdl": False,
        },
        "library": "cake_gdp_c8580552e73f2264f8eb",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": True,
            "use_initial_state": False,
            "use_state_indices": False,
        },
    },
    "cake_gdp_cbb122ec06d8e7bfbdab": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 5,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_cbb122ec06d8e7bfbdab",
        "kernel_symbol": "kernel_cake_gdp_cbb122ec06d8e7bfbdab",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 30720,
            "use_pdl": False,
        },
        "library": "cake_gdp_cbb122ec06d8e7bfbdab",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_dbd09ba202e904a29148": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "splitv",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "splitv8_tile8",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 8,
            "kind": "state_heads",
            "tile_v": 8,
        },
        "kernel": "cake_gdp_dbd09ba202e904a29148",
        "kernel_symbol": "kernel_cake_gdp_dbd09ba202e904a29148",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2304,
            "use_pdl": False,
        },
        "library": "cake_gdp_dbd09ba202e904a29148",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": 160,
            "kind": "state_heads_between",
            "rounds_at_most": 4,
        },
    },
    "cake_gdp_e1047159db983943d97b": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_e1047159db983943d97b",
        "kernel_symbol": "kernel_cake_gdp_e1047159db983943d97b",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2944,
            "use_pdl": False,
        },
        "library": "cake_gdp_e1047159db983943d97b",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 160,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 4,
                "ctas_per_sm_at_most": 64,
                "kind": "state_heads_between",
            },
            {
                "above": 160,
                "arches": ["sm_90a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_e156ee45668b2455e46d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "l8r4p",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_e156ee45668b2455e46d",
        "kernel_symbol": "kernel_cake_gdp_e156ee45668b2455e46d",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2944,
            "use_pdl": False,
        },
        "library": "cake_gdp_e156ee45668b2455e46d",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 160,
            "at_most": None,
            "ctas_per_sm_at_most": 4,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_e3643cae4c9fe94f434c": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 2,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_e3643cae4c9fe94f434c",
        "kernel_symbol": "kernel_cake_gdp_e3643cae4c9fe94f434c",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 5760,
            "use_pdl": False,
        },
        "library": "cake_gdp_e3643cae4c9fe94f434c",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_ee0cdd5d391a7f594356": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 5,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_ee0cdd5d391a7f594356",
        "kernel_symbol": "kernel_cake_gdp_ee0cdd5d391a7f594356",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 30720,
            "use_pdl": False,
        },
        "library": "cake_gdp_ee0cdd5d391a7f594356",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
                "rounds_at_most": 4,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 0.5,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 1.5,
                "ctas_per_sm_at_most": 2,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_efdf403e0df822bfc83f": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 3,
            "snapshot_noise": True,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_efdf403e0df822bfc83f",
        "kernel_symbol": "kernel_cake_gdp_efdf403e0df822bfc83f",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 61824,
            "use_pdl": False,
        },
        "library": "cake_gdp_efdf403e0df822bfc83f",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "ctas_per_sm_at_most": 3,
                "kind": "state_heads_between",
                "rounds_at_least": 10,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a"],
                "at_most": None,
                "ctas_per_sm_above": 4,
                "kind": "state_heads_between",
                "rounds_at_least": 10,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a"],
                "at_most": None,
                "ctas_per_sm_above": 3,
                "ctas_per_sm_at_most": 4,
                "kind": "state_heads_between",
                "rounds_at_least": 11,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 4,
                "ctas_per_sm_at_most": 124,
                "kind": "state_heads_between",
                "rounds_at_least": 10,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 3,
                "ctas_per_sm_at_most": 4,
                "kind": "state_heads_between",
                "rounds_at_least": 16,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 124,
                "kind": "state_heads_between",
                "rounds_at_least": 11,
                "snapshot": True,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "ctas_per_sm_at_most": 3,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
                "snapshot": True,
            },
        ],
    },
    "cake_gdp_f0b0248a8e7d3ad6047d": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_f0b0248a8e7d3ad6047d",
        "kernel_symbol": "kernel_cake_gdp_f0b0248a8e7d3ad6047d",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 17152,
            "use_pdl": False,
        },
        "library": "cake_gdp_f0b0248a8e7d3ad6047d",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_f2f0c7abbf10082a6996": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "splitv",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "splitv8_tile8",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 8,
            "kind": "state_heads",
            "tile_v": 8,
        },
        "kernel": "cake_gdp_f2f0c7abbf10082a6996",
        "kernel_symbol": "kernel_cake_gdp_f2f0c7abbf10082a6996",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2304,
            "use_pdl": False,
        },
        "library": "cake_gdp_f2f0c7abbf10082a6996",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": 160,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_f39daf3dd3cf89844fc1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90",
        "build": {
            "debug_trace": False,
            "desc_iter": False,
            "exact_diag": True,
            "inverse": "blk32",
            "io_dtype": "bf16",
            "l_tile": None,
            "mask_fast": False,
            "o_store": "st_global",
            "segment_pass": False,
            "solve_ablate": False,
            "solve_block": 4,
            "solve_ilp": 1,
            "solve_overlap": True,
            "solve_prefetch": 0,
            "split_decay": True,
            "split_items": False,
            "split_state": None,
            "split_w": None,
            "store_final_state": True,
            "unit_beta": True,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
            "wait_hint": None,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_f39daf3dd3cf89844fc1",
        "kernel_symbol": "kernel_cake_gdp_f39daf3dd3cf89844fc1",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226304,
            "use_pdl": False,
        },
        "library": "cake_gdp_f39daf3dd3cf89844fc1",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": True,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
        },
    },
    "cake_gdp_f3f05058d0b4a43f6589": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 1,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_f3f05058d0b4a43f6589",
        "kernel_symbol": "kernel_cake_gdp_f3f05058d0b4a43f6589",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 37120,
            "use_pdl": False,
        },
        "library": "cake_gdp_f3f05058d0b4a43f6589",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_f5d305e6f05978caca3c": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l4r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_f5d305e6f05978caca3c",
        "kernel_symbol": "kernel_cake_gdp_f5d305e6f05978caca3c",
        "launch": {
            "block": (
                64,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 45440,
            "use_pdl": False,
        },
        "library": "cake_gdp_f5d305e6f05978caca3c",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
                "rounds_at_most": 9,
            },
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a", "sm_107a"],
                "at_most": None,
                "kind": "state_heads_between",
                "snapshot": False,
            },
            {
                "above": 0,
                "arches": ["sm_100a", "sm_103a"],
                "at_most": None,
                "ctas_per_sm_above": 3,
                "ctas_per_sm_at_most": 4,
                "kind": "state_heads_between",
                "rounds_at_most": 10,
            },
            {
                "above": 0,
                "arches": ["sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 3,
                "ctas_per_sm_at_most": 4,
                "kind": "state_heads_between",
                "rounds_at_most": 15,
            },
            {
                "above": 0,
                "arches": ["sm_107a"],
                "at_most": None,
                "ctas_per_sm_above": 124,
                "kind": "state_heads_between",
                "rounds_at_most": 10,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "ctas_per_sm_at_most": 3,
                "kind": "state_heads_between",
                "rounds_at_most": 4,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 2,
                "ctas_per_sm_at_most": 3,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
                "snapshot": False,
            },
            {
                "above": 0,
                "arches": ["sm_90a"],
                "at_most": None,
                "ctas_per_sm_above": 3,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_f65e08f2c35db6ffe971": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 7,
            "use_qk_l2norm": True,
            "variant": "l8r2",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_f65e08f2c35db6ffe971",
        "kernel_symbol": "kernel_cake_gdp_f65e08f2c35db6ffe971",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 19968,
            "use_pdl": False,
        },
        "library": "cake_gdp_f65e08f2c35db6ffe971",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_fefa1931e0ddadb74891": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "l8r4",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 1,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_fefa1931e0ddadb74891",
        "kernel_symbol": "kernel_cake_gdp_fefa1931e0ddadb74891",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2944,
            "use_pdl": False,
        },
        "library": "cake_gdp_fefa1931e0ddadb74891",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "q": "q",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 160,
                "arches": [
                    "sm_100a",
                    "sm_103a",
                    "sm_107a",
                ],
                "at_most": None,
                "ctas_per_sm_above": 4,
                "ctas_per_sm_at_most": 64,
                "kind": "state_heads_between",
            },
            {
                "above": 160,
                "arches": [
                    "sm_90a",
                ],
                "at_most": None,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_0de8fd0ee57885709a1c": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "issue": "A3",
            "kring": 3,
            "plan": "p32_232_208",
            "ring": "qk3_o1",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_0de8fd0ee57885709a1c",
        "kernel_symbol": "kernel_cake_gdp_0de8fd0ee57885709a1c",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197376,
            "use_pdl": False,
        },
        "library": "cake_gdp_0de8fd0ee57885709a1c",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
        },
    },
    "cake_gdp_c81addff1e986f49ec99": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "issue": "A3",
            "kring": 3,
            "plan": "p32_232_208",
            "ring": "qk3_o1",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_c81addff1e986f49ec99",
        "kernel_symbol": "kernel_cake_gdp_c81addff1e986f49ec99",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197376,
            "use_pdl": False,
        },
        "library": "cake_gdp_c81addff1e986f49ec99",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": False,
        },
    },
    "cake_gdp_55a7b3c2baf155618309": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "issue": "A3",
            "kring": 3,
            "plan": "p32_232_208",
            "ring": "qk3_o1",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_55a7b3c2baf155618309",
        "kernel_symbol": "kernel_cake_gdp_55a7b3c2baf155618309",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197376,
            "use_pdl": False,
        },
        "library": "cake_gdp_55a7b3c2baf155618309",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_5acbda2bd1002fd68a95": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "csr": True,
            "issue": "A3",
            "kring": 3,
            "plan": "p32_240_232",
            "plan_guard": True,
            "ring": "qk3_o1",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_5acbda2bd1002fd68a95",
        "kernel_symbol": "kernel_cake_gdp_5acbda2bd1002fd68a95",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198528,
            "use_pdl": False,
        },
        "library": "cake_gdp_5acbda2bd1002fd68a95",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": False,
            "use_state_indices": False,
        },
    },
    "cake_gdp_edfb71d95c7a9c2a133d": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "csr": True,
            "issue": "A3",
            "kring": 3,
            "plan": "p32_240_232",
            "plan_guard": True,
            "ring": "qk3_o1",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_edfb71d95c7a9c2a133d",
        "kernel_symbol": "kernel_cake_gdp_edfb71d95c7a9c2a133d",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198528,
            "use_pdl": False,
        },
        "library": "cake_gdp_edfb71d95c7a9c2a133d",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": False,
        },
    },
    "cake_gdp_446ecf26f72f95aacf84": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "csr": True,
            "issue": "A3",
            "kring": 3,
            "plan": "p32_240_232",
            "plan_guard": True,
            "ring": "qk3_o1",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_446ecf26f72f95aacf84",
        "kernel_symbol": "kernel_cake_gdp_446ecf26f72f95aacf84",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198528,
            "use_pdl": False,
        },
        "library": "cake_gdp_446ecf26f72f95aacf84",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_6108f3bb9a5231a2ccab": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "issue": "A3",
            "kring": 3,
            "plan": "p32_232_208",
            "ring": "qk3_o1",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_6108f3bb9a5231a2ccab",
        "kernel_symbol": "kernel_cake_gdp_6108f3bb9a5231a2ccab",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197376,
            "use_pdl": False,
        },
        "library": "cake_gdp_6108f3bb9a5231a2ccab",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_31952c28b9b7f7594e32": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "issue": "A3",
            "kring": 3,
            "plan": "p32_232_208",
            "ring": "qk3_o1",
            "segment_pass": False,
            "state_dtype": "bf16",
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_31952c28b9b7f7594e32",
        "kernel_symbol": "kernel_cake_gdp_31952c28b9b7f7594e32",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197376,
            "use_pdl": False,
        },
        "library": "cake_gdp_31952c28b9b7f7594e32",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_90c40c9a853e217d9464": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "issue": "A3",
            "kring": 3,
            "plan": "p32_232_208",
            "ring": "qk3_o1",
            "segment_pass": False,
            "state_dtype": "f16",
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_90c40c9a853e217d9464",
        "kernel_symbol": "kernel_cake_gdp_90c40c9a853e217d9464",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197376,
            "use_pdl": False,
        },
        "library": "cake_gdp_90c40c9a853e217d9464",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_d8969198c1f1eb037c80": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "csr": True,
            "issue": "A3",
            "kring": 3,
            "plan": "p32_240_232",
            "plan_guard": True,
            "ring": "qk3_o1",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_d8969198c1f1eb037c80",
        "kernel_symbol": "kernel_cake_gdp_d8969198c1f1eb037c80",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198528,
            "use_pdl": False,
        },
        "library": "cake_gdp_d8969198c1f1eb037c80",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_334cfa42025043287dfa": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "csr": True,
            "issue": "A3",
            "kring": 3,
            "plan": "p32_240_232",
            "plan_guard": True,
            "ring": "qk3_o1",
            "segment_pass": False,
            "split_items": True,
            "state_dtype": "bf16",
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_334cfa42025043287dfa",
        "kernel_symbol": "kernel_cake_gdp_334cfa42025043287dfa",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198528,
            "use_pdl": False,
        },
        "library": "cake_gdp_334cfa42025043287dfa",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_1e2a7c403e001992a9ab": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "Q",
            ],
            [
                "buffer",
                "K",
            ],
            [
                "buffer",
                "V",
            ],
            [
                "buffer",
                "O",
            ],
            [
                "buffer",
                "gate",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "cu_seqlens",
            ],
            [
                "buffer",
                "state_indices",
            ],
            [
                "buffer",
                "output_slot_indices",
            ],
            [
                "buffer",
                "initial_state",
            ],
            [
                "buffer",
                "output_state",
            ],
            [
                "buffer",
                "tensormap_workspace",
            ],
            [
                "parameter",
                "initial_state_stride_slot",
            ],
            [
                "parameter",
                "output_state_stride_slot",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "num_seqs",
            ],
            [
                "parameter",
                "total_tokens",
            ],
            [
                "parameter",
                "num_heads",
            ],
            [
                "parameter",
                "num_householder",
            ],
            [
                "parameter",
                "total_tiles",
            ],
            [
                "parameter",
                "metadata_i64_mask",
            ],
            [
                "buffer",
                "ainv_workspace",
            ],
            [
                "parameter",
                "ainv_chunk_stride",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "prefill_sm90_v6",
        "build": {
            "arm": "fork_exact_serial_wl_abo_sl4p_fk_pb_ogu",
            "csr": True,
            "issue": "A3",
            "kring": 3,
            "plan": "p32_240_232",
            "plan_guard": True,
            "ring": "qk3_o1",
            "segment_pass": False,
            "split_items": True,
            "state_dtype": "f16",
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_1e2a7c403e001992a9ab",
        "kernel_symbol": "kernel_cake_gdp_1e2a7c403e001992a9ab",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 198528,
            "use_pdl": False,
        },
        "library": "cake_gdp_1e2a7c403e001992a9ab",
        "logical": {
            "K": "k",
            "O": "o",
            "Q": "q",
            "V": "v",
            "ainv_chunk_stride": "ainv_chunk_stride",
            "ainv_workspace": "ainv_workspace",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "metadata_i64_mask": "metadata_i64_mask",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_seqs": "num_seqs",
            "output_slot_indices": "output_slot_indices",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "state_indices": "state_indices",
            "tensormap_workspace": "tensormap_workspace",
            "total_tiles": "total_tiles",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "prefill_flags",
            "segment_pass": False,
            "split_items": True,
            "store_final_state": True,
            "unit_beta": False,
            "unit_gates": False,
            "use_initial_state": True,
            "use_output_state_indices": True,
            "use_state_indices": True,
        },
    },
    "cake_gdp_ff55032bc5fdf2697d77": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "splitv",
        "build": {
            "draw_share": True,
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 1,
            "use_qk_l2norm": True,
            "variant": "splitv8_tile8",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 8,
            "kind": "state_heads",
            "tile_v": 8,
        },
        "kernel": "cake_gdp_ff55032bc5fdf2697d77",
        "kernel_symbol": "kernel_cake_gdp_ff55032bc5fdf2697d77",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 2304,
            "use_pdl": False,
        },
        "library": "cake_gdp_ff55032bc5fdf2697d77",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": 160,
            "kind": "state_heads_between",
            "rounds_at_least": 5,
        },
    },
    "cake_gdp_4c574be45e595fe98665": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 2,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_4c574be45e595fe98665",
        "kernel_symbol": "kernel_cake_gdp_4c574be45e595fe98665",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 13952,
            "use_pdl": False,
        },
        "library": "cake_gdp_4c574be45e595fe98665",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_db30017cc306551537d8": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 3,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_db30017cc306551537d8",
        "kernel_symbol": "kernel_cake_gdp_db30017cc306551537d8",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 16768,
            "use_pdl": False,
        },
        "library": "cake_gdp_db30017cc306551537d8",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2.5,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_03fc4d822d2896c11df9": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 4,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_03fc4d822d2896c11df9",
        "kernel_symbol": "kernel_cake_gdp_03fc4d822d2896c11df9",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 19584,
            "use_pdl": False,
        },
        "library": "cake_gdp_03fc4d822d2896c11df9",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 0.5,
                "ctas_per_sm_at_most": 1,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 1,
                "ctas_per_sm_at_most": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_2bbb0a6eea719d10ec5c": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 5,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_2bbb0a6eea719d10ec5c",
        "kernel_symbol": "kernel_cake_gdp_2bbb0a6eea719d10ec5c",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 22528,
            "use_pdl": False,
        },
        "library": "cake_gdp_2bbb0a6eea719d10ec5c",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
                "rounds_at_least": 5,
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 1,
                "ctas_per_sm_at_most": 1.5,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 2,
                "ctas_per_sm_at_most": 2.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_c26823b0a9a42e400ad0": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 6,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_c26823b0a9a42e400ad0",
        "kernel_symbol": "kernel_cake_gdp_c26823b0a9a42e400ad0",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 25344,
            "use_pdl": False,
        },
        "library": "cake_gdp_c26823b0a9a42e400ad0",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 1,
                "ctas_per_sm_at_most": 1.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_e506863593932b5cf7d2": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 7,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_e506863593932b5cf7d2",
        "kernel_symbol": "kernel_cake_gdp_e506863593932b5cf7d2",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 28160,
            "use_pdl": False,
        },
        "library": "cake_gdp_e506863593932b5cf7d2",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 1,
                "ctas_per_sm_at_most": 1.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_a5f2411d386e80d6c500": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 8,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_a5f2411d386e80d6c500",
        "kernel_symbol": "kernel_cake_gdp_a5f2411d386e80d6c500",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 30976,
            "use_pdl": False,
        },
        "library": "cake_gdp_a5f2411d386e80d6c500",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 1,
                "ctas_per_sm_at_most": 1.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_0e9af5fb535caec5f960": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 8,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": 16,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_0e9af5fb535caec5f960",
        "kernel_symbol": "kernel_cake_gdp_0e9af5fb535caec5f960",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 53632,
            "use_pdl": False,
        },
        "library": "cake_gdp_0e9af5fb535caec5f960",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": [
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_at_most": 0.5,
                "kind": "state_heads_between",
            },
            {
                "above": 0,
                "at_most": None,
                "ctas_per_sm_above": 1,
                "ctas_per_sm_at_most": 1.5,
                "kind": "state_heads_between",
            },
        ],
    },
    "cake_gdp_527260b8ec384f213efd": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 1,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_527260b8ec384f213efd",
        "kernel_symbol": "kernel_cake_gdp_527260b8ec384f213efd",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 28928,
            "use_pdl": False,
        },
        "library": "cake_gdp_527260b8ec384f213efd",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2.5,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_09db3ffab8d0378caa18": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 2,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_09db3ffab8d0378caa18",
        "kernel_symbol": "kernel_cake_gdp_09db3ffab8d0378caa18",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 41344,
            "use_pdl": False,
        },
        "library": "cake_gdp_09db3ffab8d0378caa18",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2.5,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_ce8bf154da25575debfc": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 3,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_ce8bf154da25575debfc",
        "kernel_symbol": "kernel_cake_gdp_ce8bf154da25575debfc",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 53632,
            "use_pdl": False,
        },
        "library": "cake_gdp_ce8bf154da25575debfc",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_1654ec2bd9e2b4374ad9": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            [
                "buffer",
                "q",
            ],
            [
                "buffer",
                "k",
            ],
            [
                "buffer",
                "v",
            ],
            [
                "buffer",
                "state",
            ],
            [
                "buffer",
                "A_log",
            ],
            [
                "buffer",
                "a",
            ],
            [
                "buffer",
                "dt_bias",
            ],
            [
                "buffer",
                "b",
            ],
            [
                "buffer",
                "out",
            ],
            [
                "buffer",
                "initial_state_indices",
            ],
            [
                "buffer",
                "output_state_indices",
            ],
            [
                "buffer",
                "ssm_state_indices",
            ],
            [
                "buffer",
                "rand_seed",
            ],
            [
                "parameter",
                "heads",
            ],
            [
                "parameter",
                "num_tokens",
            ],
            [
                "parameter",
                "scale",
            ],
            [
                "parameter",
                "disable_state_update",
            ],
            [
                "parameter",
                "use_ssm_state_indices",
            ],
            [
                "parameter",
                "philox_rounds",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "body": "lanerow",
        "build": {
            "max_t_steps": 16,
            "num_householder": 4,
            "state_dtype": "f16",
            "t_steps": None,
            "use_qk_l2norm": True,
            "variant": "l8r2h",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_state": 2,
            "kind": "state_heads",
        },
        "kernel": "cake_gdp_1654ec2bd9e2b4374ad9",
        "kernel_symbol": "kernel_cake_gdp_1654ec2bd9e2b4374ad9",
        "launch": {
            "block": (
                128,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 66048,
            "use_pdl": False,
        },
        "library": "cake_gdp_1654ec2bd9e2b4374ad9",
        "logical": {
            "A_log": "A_log",
            "a": "a",
            "b": "b",
            "disable_state_update": "disable_state_update",
            "dt_bias": "dt_bias",
            "heads": "heads",
            "initial_state_indices": "initial_state_indices",
            "k": "k",
            "num_tokens": "num_tokens",
            "out": "out",
            "output_state_indices": "output_state_indices",
            "philox_rounds": "philox_rounds",
            "q": "q",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "ssm_state_indices": "ssm_state_indices",
            "state": "state",
            "use_ssm_state_indices": "use_ssm_state_indices",
            "v": "v",
        },
        "role": "decode",
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 1.5,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_157b3ebf33cf7c928af5": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_157b3ebf33cf7c928af5",
        "kernel_symbol": "kernel_cake_gdp_157b3ebf33cf7c928af5",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_157b3ebf33cf7c928af5",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_68796be6ca9ae12f2e19": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_68796be6ca9ae12f2e19",
        "kernel_symbol": "kernel_cake_gdp_68796be6ca9ae12f2e19",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_68796be6ca9ae12f2e19",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_26a79828ab8713b466c5": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_26a79828ab8713b466c5",
        "kernel_symbol": "kernel_cake_gdp_26a79828ab8713b466c5",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_26a79828ab8713b466c5",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_04160875e0e65fe60026": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_04160875e0e65fe60026",
        "kernel_symbol": "kernel_cake_gdp_04160875e0e65fe60026",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_04160875e0e65fe60026",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_f52e7ec41bb7f27cd856": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_f52e7ec41bb7f27cd856",
        "kernel_symbol": "kernel_cake_gdp_f52e7ec41bb7f27cd856",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_f52e7ec41bb7f27cd856",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_4f440244f31edea6a417": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_4f440244f31edea6a417",
        "kernel_symbol": "kernel_cake_gdp_4f440244f31edea6a417",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_4f440244f31edea6a417",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_f40413c9ea5f77600699": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_f40413c9ea5f77600699",
        "kernel_symbol": "kernel_cake_gdp_f40413c9ea5f77600699",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_f40413c9ea5f77600699",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_c39ce0efb75d6240b1d4": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_c39ce0efb75d6240b1d4",
        "kernel_symbol": "kernel_cake_gdp_c39ce0efb75d6240b1d4",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_c39ce0efb75d6240b1d4",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_63289a15605137af5bf8": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "state_dtype": "bf16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_63289a15605137af5bf8",
        "kernel_symbol": "kernel_cake_gdp_63289a15605137af5bf8",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_63289a15605137af5bf8",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_591de83b8e98709df3f3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "state_dtype": "bf16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_591de83b8e98709df3f3",
        "kernel_symbol": "kernel_cake_gdp_591de83b8e98709df3f3",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_591de83b8e98709df3f3",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_a12e8746d7541ad471ac": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "state_dtype": "bf16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_a12e8746d7541ad471ac",
        "kernel_symbol": "kernel_cake_gdp_a12e8746d7541ad471ac",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_a12e8746d7541ad471ac",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_326b3128513266c6f402": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "state_dtype": "bf16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_326b3128513266c6f402",
        "kernel_symbol": "kernel_cake_gdp_326b3128513266c6f402",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_326b3128513266c6f402",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_1ffa9f43b2424a9014e7": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "state_dtype": "f16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_1ffa9f43b2424a9014e7",
        "kernel_symbol": "kernel_cake_gdp_1ffa9f43b2424a9014e7",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_1ffa9f43b2424a9014e7",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_241bf09e6734085e173e": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+vyh+vyp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "state_dtype": "f16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_241bf09e6734085e173e",
        "kernel_symbol": "kernel_cake_gdp_241bf09e6734085e173e",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_241bf09e6734085e173e",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
    "cake_gdp_c6c640ba25906f3b851a": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag+tailc",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": False,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": False,
            "split_thr": -48.0,
            "split_warmup": 8,
            "stamps": "none",
            "state_dtype": "f16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_c6c640ba25906f3b851a",
        "kernel_symbol": "kernel_cake_gdp_c6c640ba25906f3b851a",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_c6c640ba25906f3b851a",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": False,
        },
    },
    "cake_gdp_13b07ddf43faaf58b7d1": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "Omap"],
            ["buffer", "tmap_ws"],
            ["buffer", "gate"],
            ["buffer", "beta"],
            ["buffer", "cu_seqlens"],
            ["buffer", "items"],
            ["buffer", "initial_state"],
            ["buffer", "output_state"],
            ["parameter", "state_stride_slot"],
            ["parameter", "scale"],
            ["parameter", "num_items"],
            ["parameter", "total_tokens"],
            ["parameter", "num_heads"],
            ["parameter", "num_householder"],
            ["parameter", "flags"],
            ["buffer", "split_ws"],
            ["parameter", "num_seqs"],
            ["buffer", "state_indices"],
            ["buffer", "output_state_indices"],
            ["parameter", "initial_state_stride_slot"],
            ["parameter", "output_state_stride_slot"],
            ["buffer", "rand_seed"],
            ["parameter", "philox_rounds"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "body": "prefill_bw_v6",
        "build": {
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+yh+yp+bdiag",
            "arm": "full",
            "cg0": "lock",
            "cp": False,
            "csr": True,
            "diet": "issuer_compact+roll_ah",
            "dispatch": False,
            "du_split_issue": False,
            "fused": "m128",
            "gate_tab": True,
            "householders": 3,
            "inverse": "c",
            "iss_order": "m3m4",
            "ostore": "tma",
            "plan_guard": True,
            "regs": None,
            "scan": "iii",
            "split_du": False,
            "split_items": True,
            "split_thr": -14.43,
            "split_warmup": 128,
            "stamps": "none",
            "state_dtype": "f16",
            "use_output_state_indices": True,
            "use_state_indices": True,
            "vsplit": False,
            "with_beta": True,
            "xh": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
        },
        "kernel": "cake_gdp_13b07ddf43faaf58b7d1",
        "kernel_symbol": "kernel_cake_gdp_13b07ddf43faaf58b7d1",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_13b07ddf43faaf58b7d1",
        "logical": {
            "K": "k",
            "O": "o",
            "Omap": "o",
            "Q": "q",
            "V": "v",
            "beta": "beta",
            "cu_seqlens": "cu_seqlens",
            "flags": "flags",
            "gate": "gate",
            "initial_state": "initial_state",
            "initial_state_stride_slot": "initial_state_stride_slot",
            "items": "items",
            "num_heads": "num_heads",
            "num_householder": "num_householder",
            "num_items": "num_items",
            "num_seqs": "num_seqs",
            "output_state": "output_state",
            "output_state_indices": "output_state_indices",
            "output_state_stride_slot": "output_state_stride_slot",
            "philox_rounds": "philox_rounds",
            "rand_seed": "rand_seed",
            "scale": "scale",
            "split_ws": "split_workspace",
            "state_indices": "state_indices",
            "state_stride_slot": "state_stride_slot",
            "tmap_ws": "tensormap_workspace",
            "total_tokens": "total_tokens",
        },
        "role": "prefill",
        "select_rule": {
            "kind": "v6_launch",
            "split_items": True,
        },
    },
}
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_90a": {
        "decode_nh1": [
            "cake_gdp_78fc9caafee16e2af047",
            "cake_gdp_f3f05058d0b4a43f6589",
            "cake_gdp_527260b8ec384f213efd",
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
            "cake_gdp_aa7f7810c60fe76c14a6",
            "cake_gdp_09db3ffab8d0378caa18",
        ],
        "decode_nh3": [
            "cake_gdp_068a895d0c8fa77ed2f9",
            "cake_gdp_fefa1931e0ddadb74891",
            "cake_gdp_e3643cae4c9fe94f434c",
            "cake_gdp_159ecba6294f1c73ed91",
            "cake_gdp_27eb88658ee4bdc4ae35",
            "cake_gdp_baca2c2d21cf59fc2d3c",
            "cake_gdp_32a3f37f5ef3015142a5",
            "cake_gdp_46ef1a68db53b4c532e9",
            "cake_gdp_f0b0248a8e7d3ad6047d",
            "cake_gdp_a226c899893599e679a3",
            "cake_gdp_f65e08f2c35db6ffe971",
            "cake_gdp_1a3d01a5826508d64a06",
            "cake_gdp_3cbdef34ce5b580c2216",
            "cake_gdp_5970bfa6676087a3ce92",
            "cake_gdp_450bb0db3b79acafa419",
            "cake_gdp_c51f69d9ac126bbaab36",
            "cake_gdp_585a5ea4cbdd07f1b2a8",
            "cake_gdp_8f8062d89b8b871f1e2f",
            "cake_gdp_6bb51453cd15a43da9c2",
            "cake_gdp_bc76e12e0c247b44a27d",
            "cake_gdp_5bc4ef21b6e7aefed05f",
            "cake_gdp_ee0cdd5d391a7f594356",
            "cake_gdp_7e4385817ba59915bc61",
            "cake_gdp_8e26e2c2d0cb0a7fbee9",
            "cake_gdp_cbb122ec06d8e7bfbdab",
            "cake_gdp_f5d305e6f05978caca3c",
            "cake_gdp_efdf403e0df822bfc83f",
            "cake_gdp_bb55d7d5e4db0d95acce",
            "cake_gdp_11085a9519b214993653",
            "cake_gdp_8aabe827188edf904960",
            "cake_gdp_b245506c43aed8c3e6b4",
            "cake_gdp_16d24c9e19fa73fe57ee",
            "cake_gdp_b047f2f1a20630e6f36b",
            "cake_gdp_c3cf32174475946d9c35",
            "cake_gdp_9bfd49d6bbbb3d849f30",
            "cake_gdp_72b6872c98ed9abf0bdf",
            "cake_gdp_e1047159db983943d97b",
            "cake_gdp_4c574be45e595fe98665",
            "cake_gdp_db30017cc306551537d8",
            "cake_gdp_03fc4d822d2896c11df9",
            "cake_gdp_2bbb0a6eea719d10ec5c",
            "cake_gdp_c26823b0a9a42e400ad0",
            "cake_gdp_e506863593932b5cf7d2",
            "cake_gdp_a5f2411d386e80d6c500",
            "cake_gdp_0e9af5fb535caec5f960",
            "cake_gdp_ce8bf154da25575debfc",
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
            "cake_gdp_717ce3c684b41c37e673",
            "cake_gdp_1654ec2bd9e2b4374ad9",
        ],
        "prefill": [
            "cake_gdp_0de8fd0ee57885709a1c",
            "cake_gdp_c81addff1e986f49ec99",
            "cake_gdp_55a7b3c2baf155618309",
            "cake_gdp_c8580552e73f2264f8eb",
            "cake_gdp_f39daf3dd3cf89844fc1",
            "cake_gdp_3220d37914678385a837",
            "cake_gdp_245414df8449fb1825a1",
            "cake_gdp_b81854850bf4b769295c",
            "cake_gdp_5acbda2bd1002fd68a95",
            "cake_gdp_edfb71d95c7a9c2a133d",
            "cake_gdp_446ecf26f72f95aacf84",
            "cake_gdp_6108f3bb9a5231a2ccab",
            "cake_gdp_31952c28b9b7f7594e32",
            "cake_gdp_90c40c9a853e217d9464",
            "cake_gdp_d8969198c1f1eb037c80",
            "cake_gdp_334cfa42025043287dfa",
            "cake_gdp_1e2a7c403e001992a9ab",
        ],
    },
    "sm_100a": {
        "decode_nh1": [
            "cake_gdp_78fc9caafee16e2af047",
            "cake_gdp_f3f05058d0b4a43f6589",
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
            "cake_gdp_aa7f7810c60fe76c14a6",
        ],
        "decode_nh3": [
            "cake_gdp_f2f0c7abbf10082a6996",
            "cake_gdp_e156ee45668b2455e46d",
            "cake_gdp_4d40c98af90453032200",
            "cake_gdp_fefa1931e0ddadb74891",
            "cake_gdp_e3643cae4c9fe94f434c",
            "cake_gdp_159ecba6294f1c73ed91",
            "cake_gdp_27eb88658ee4bdc4ae35",
            "cake_gdp_32a3f37f5ef3015142a5",
            "cake_gdp_46ef1a68db53b4c532e9",
            "cake_gdp_f0b0248a8e7d3ad6047d",
            "cake_gdp_a226c899893599e679a3",
            "cake_gdp_f65e08f2c35db6ffe971",
            "cake_gdp_1a3d01a5826508d64a06",
            "cake_gdp_3cbdef34ce5b580c2216",
            "cake_gdp_5970bfa6676087a3ce92",
            "cake_gdp_450bb0db3b79acafa419",
            "cake_gdp_c51f69d9ac126bbaab36",
            "cake_gdp_585a5ea4cbdd07f1b2a8",
            "cake_gdp_dbd09ba202e904a29148",
            "cake_gdp_8f4ad44e524af3f3cbb9",
            "cake_gdp_0f9d4678110f025383b2",
            "cake_gdp_e1047159db983943d97b",
            "cake_gdp_c3cf32174475946d9c35",
            "cake_gdp_6bb51453cd15a43da9c2",
            "cake_gdp_5bc4ef21b6e7aefed05f",
            "cake_gdp_ee0cdd5d391a7f594356",
            "cake_gdp_cbb122ec06d8e7bfbdab",
            "cake_gdp_9bfd49d6bbbb3d849f30",
            "cake_gdp_0c31b01fa2a76d426c5f",
            "cake_gdp_72b6872c98ed9abf0bdf",
            "cake_gdp_bc76e12e0c247b44a27d",
            "cake_gdp_7e4385817ba59915bc61",
            "cake_gdp_bb55d7d5e4db0d95acce",
            "cake_gdp_8e26e2c2d0cb0a7fbee9",
            "cake_gdp_16d24c9e19fa73fe57ee",
            "cake_gdp_f5d305e6f05978caca3c",
            "cake_gdp_b047f2f1a20630e6f36b",
            "cake_gdp_efdf403e0df822bfc83f",
            "cake_gdp_ff55032bc5fdf2697d77",
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
            "cake_gdp_717ce3c684b41c37e673",
        ],
        "prefill": [
            "cake_gdp_157b3ebf33cf7c928af5",
            "cake_gdp_68796be6ca9ae12f2e19",
            "cake_gdp_f52e7ec41bb7f27cd856",
            "cake_gdp_4f440244f31edea6a417",
            "cake_gdp_63289a15605137af5bf8",
            "cake_gdp_591de83b8e98709df3f3",
            "cake_gdp_1ffa9f43b2424a9014e7",
            "cake_gdp_241bf09e6734085e173e",
        ],
    },
    "sm_103a": {
        "decode_nh1": [
            "cake_gdp_78fc9caafee16e2af047",
            "cake_gdp_f3f05058d0b4a43f6589",
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
            "cake_gdp_aa7f7810c60fe76c14a6",
        ],
        "decode_nh3": [
            "cake_gdp_f2f0c7abbf10082a6996",
            "cake_gdp_e156ee45668b2455e46d",
            "cake_gdp_4d40c98af90453032200",
            "cake_gdp_fefa1931e0ddadb74891",
            "cake_gdp_e3643cae4c9fe94f434c",
            "cake_gdp_159ecba6294f1c73ed91",
            "cake_gdp_27eb88658ee4bdc4ae35",
            "cake_gdp_32a3f37f5ef3015142a5",
            "cake_gdp_46ef1a68db53b4c532e9",
            "cake_gdp_f0b0248a8e7d3ad6047d",
            "cake_gdp_a226c899893599e679a3",
            "cake_gdp_f65e08f2c35db6ffe971",
            "cake_gdp_1a3d01a5826508d64a06",
            "cake_gdp_3cbdef34ce5b580c2216",
            "cake_gdp_5970bfa6676087a3ce92",
            "cake_gdp_450bb0db3b79acafa419",
            "cake_gdp_c51f69d9ac126bbaab36",
            "cake_gdp_585a5ea4cbdd07f1b2a8",
            "cake_gdp_dbd09ba202e904a29148",
            "cake_gdp_8f4ad44e524af3f3cbb9",
            "cake_gdp_0f9d4678110f025383b2",
            "cake_gdp_e1047159db983943d97b",
            "cake_gdp_c3cf32174475946d9c35",
            "cake_gdp_6bb51453cd15a43da9c2",
            "cake_gdp_5bc4ef21b6e7aefed05f",
            "cake_gdp_ee0cdd5d391a7f594356",
            "cake_gdp_cbb122ec06d8e7bfbdab",
            "cake_gdp_9bfd49d6bbbb3d849f30",
            "cake_gdp_0c31b01fa2a76d426c5f",
            "cake_gdp_72b6872c98ed9abf0bdf",
            "cake_gdp_bc76e12e0c247b44a27d",
            "cake_gdp_7e4385817ba59915bc61",
            "cake_gdp_bb55d7d5e4db0d95acce",
            "cake_gdp_8e26e2c2d0cb0a7fbee9",
            "cake_gdp_16d24c9e19fa73fe57ee",
            "cake_gdp_f5d305e6f05978caca3c",
            "cake_gdp_b047f2f1a20630e6f36b",
            "cake_gdp_efdf403e0df822bfc83f",
            "cake_gdp_ff55032bc5fdf2697d77",
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
            "cake_gdp_717ce3c684b41c37e673",
        ],
        "prefill": [
            "cake_gdp_26a79828ab8713b466c5",
            "cake_gdp_04160875e0e65fe60026",
            "cake_gdp_f40413c9ea5f77600699",
            "cake_gdp_c39ce0efb75d6240b1d4",
            "cake_gdp_a12e8746d7541ad471ac",
            "cake_gdp_326b3128513266c6f402",
            "cake_gdp_c6c640ba25906f3b851a",
            "cake_gdp_13b07ddf43faaf58b7d1",
        ],
    },
    "sm_107a": {
        "decode_nh1": [
            "cake_gdp_78fc9caafee16e2af047",
            "cake_gdp_f3f05058d0b4a43f6589",
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
            "cake_gdp_aa7f7810c60fe76c14a6",
        ],
        "decode_nh3": [
            "cake_gdp_f2f0c7abbf10082a6996",
            "cake_gdp_e156ee45668b2455e46d",
            "cake_gdp_4d40c98af90453032200",
            "cake_gdp_fefa1931e0ddadb74891",
            "cake_gdp_e3643cae4c9fe94f434c",
            "cake_gdp_159ecba6294f1c73ed91",
            "cake_gdp_27eb88658ee4bdc4ae35",
            "cake_gdp_32a3f37f5ef3015142a5",
            "cake_gdp_46ef1a68db53b4c532e9",
            "cake_gdp_f0b0248a8e7d3ad6047d",
            "cake_gdp_a226c899893599e679a3",
            "cake_gdp_f65e08f2c35db6ffe971",
            "cake_gdp_1a3d01a5826508d64a06",
            "cake_gdp_3cbdef34ce5b580c2216",
            "cake_gdp_5970bfa6676087a3ce92",
            "cake_gdp_450bb0db3b79acafa419",
            "cake_gdp_c51f69d9ac126bbaab36",
            "cake_gdp_585a5ea4cbdd07f1b2a8",
            "cake_gdp_dbd09ba202e904a29148",
            "cake_gdp_8f4ad44e524af3f3cbb9",
            "cake_gdp_0f9d4678110f025383b2",
            "cake_gdp_e1047159db983943d97b",
            "cake_gdp_c3cf32174475946d9c35",
            "cake_gdp_6bb51453cd15a43da9c2",
            "cake_gdp_5bc4ef21b6e7aefed05f",
            "cake_gdp_ee0cdd5d391a7f594356",
            "cake_gdp_cbb122ec06d8e7bfbdab",
            "cake_gdp_9bfd49d6bbbb3d849f30",
            "cake_gdp_0c31b01fa2a76d426c5f",
            "cake_gdp_72b6872c98ed9abf0bdf",
            "cake_gdp_bc76e12e0c247b44a27d",
            "cake_gdp_7e4385817ba59915bc61",
            "cake_gdp_bb55d7d5e4db0d95acce",
            "cake_gdp_8e26e2c2d0cb0a7fbee9",
            "cake_gdp_16d24c9e19fa73fe57ee",
            "cake_gdp_f5d305e6f05978caca3c",
            "cake_gdp_b047f2f1a20630e6f36b",
            "cake_gdp_efdf403e0df822bfc83f",
            "cake_gdp_ff55032bc5fdf2697d77",
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
            "cake_gdp_717ce3c684b41c37e673",
        ],
    },
}

# One verified build per architecture: H100 / H200 (``sm_90a``), B200 / GB200
# (``sm_100a``), B300 / GB300 (``sm_103a``) and R200 / VR200 (``sm_107a``; the
# programs and bands of ``sm_100a``).  Every architecture is compiled
# with its exact ``-gencode`` flag set (``build.json`` ``arch_flags``); there is
# no family fallback.
ARCHES = ("sm_90a", "sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
HEAD_SIZE_K = 128
HEAD_SIZE_V = 64
NUM_HOUSEHOLDER = (1, 2, 3, 4)
# The FP32 state pool is read with 16-byte asynchronous copies: a 16-byte aligned
# base and strides that are multiples of four elements.  One-line switch: a
# 256-bit-store program raises these to 32 bytes / eight elements.
STATE_ALIGNMENT_BYTES = 16
STATE_STRIDE_ELEMENTS = 4
# The FP16 state pool is read and written in 8-byte pieces of four elements: an
# 8-byte aligned base and the same strides.
F16_STATE_ALIGNMENT_BYTES = 8
# The state pool dtypes of the decode records, as their build spells them
# (``build["state_dtype"]``; an FP32 record spells none).
STATE_DTYPES: dict[torch.dtype, str] = {torch.float32: "f32", torch.float16: "f16"}
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "state",
    "A_log",
    "a",
    "dt_bias",
    "b",
    "out",
    "initial_state_indices",
    "output_state_indices",
    "ssm_state_indices",
    "rand_seed",
)
LOGICAL_SCALARS = (
    "heads",
    "num_tokens",
    "scale",
    "disable_state_update",
    "use_ssm_state_indices",
    "philox_rounds",
)


def route_key(num_householder: int, use_qk_l2norm: bool = True) -> str:
    """Name the route of one public call: ``decode_nh<n>`` (``_raw`` without the fused q / k L2 normalization)."""

    n = int(num_householder)
    if n not in NUM_HOUSEHOLDER:
        raise ValueError(
            f"num_householder must be one of {NUM_HOUSEHOLDER}, got {num_householder!r}"
        )
    return f"decode_nh{n}" + ("" if use_qk_l2norm else "_raw")


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def state_dtype_key(state_dtype: torch.dtype) -> str:
    """The records' spelling of a state pool dtype (``build["state_dtype"]``): ``"f32"`` or ``"f16"``."""

    key = STATE_DTYPES.get(state_dtype)
    if key is None:
        raise ValueError(
            f"the Cake GDP decode state pool is one of {list(STATE_DTYPES)}, got {state_dtype!r}"
        )
    return key


def _route_names(
    arch: str, num_householder: int, use_qk_l2norm: bool, state_dtype: torch.dtype
) -> tuple[str, ...]:
    """The records of one route on ``arch`` whose state pool is ``state_dtype``, filtered once per argument set
    (cached: an eager call pays no filtering).  ``num_householder`` and ``use_qk_l2norm`` are cached as ``int`` and
    ``bool``, so a NumPy or torch scalar of the same value reuses that entry."""

    return _route_names_cached(
        arch, int(num_householder), bool(use_qk_l2norm), state_dtype
    )


@functools.cache
def _route_names_cached(
    arch: str, num_householder: int, use_qk_l2norm: bool, state_dtype: torch.dtype
) -> tuple[str, ...]:
    key = state_dtype_key(state_dtype)
    names = ROUTES.get(arch, {}).get(route_key(num_householder, use_qk_l2norm), ())
    return tuple(
        name
        for name in names
        if MODULES[name]["build"].get("state_dtype", "f32") == key
    )


def supported_num_householder(
    arch: str, use_qk_l2norm: bool = True, *, state_dtype: torch.dtype = torch.float32
) -> tuple[int, ...]:
    """The ``n`` values with an exported route on ``arch`` for a ``state_dtype`` state pool."""

    return tuple(
        n for n in NUM_HOUSEHOLDER if _route_names(arch, n, use_qk_l2norm, state_dtype)
    )


def max_num_tokens(
    arch: str,
    num_householder: int,
    use_qk_l2norm: bool = True,
    *,
    state_dtype: torch.dtype = torch.float32,
) -> int:
    """The largest token count ``T`` an exported program of the route admits for a ``state_dtype`` state pool (0
    when the route has none)."""

    names = _route_names(arch, num_householder, use_qk_l2norm, state_dtype)
    bound = 0
    for name in names:
        build = MODULES[name]["build"]
        bound = max(
            bound,
            build["t_steps"] if build["t_steps"] is not None else build["max_t_steps"],
        )
    return bound


def route_applies(
    *,
    device_capability: Sequence[int],
    num_householder: int,
    num_tokens: int,
    use_qk_l2norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
) -> bool:
    """Whether the verified export owns one decode call on a ``state_dtype`` state pool (``False`` for a dtype
    outside :data:`STATE_DTYPES`, as the package's ``is_cake_gdp_supported`` answers)."""

    arch = arch_for_capability(device_capability)
    if (
        arch is None
        or int(num_householder) not in NUM_HOUSEHOLDER
        or state_dtype not in tuple(STATE_DTYPES)
    ):
        return False
    return (
        1
        <= int(num_tokens)
        <= max_num_tokens(arch, num_householder, use_qk_l2norm, state_dtype=state_dtype)
    )


def route_modules(
    arch: str, num_householder: int, use_qk_l2norm: bool = True
) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(num_householder, use_qk_l2norm)
    names = ROUTES.get(arch, {}).get(key)
    if not names:
        raise ValueError(
            f"unsupported Cake GDP decode route {key!r} on {arch!r}; exported routes on this architecture: "
            f"{sorted(ROUTES.get(arch, {}))}"
        )
    return tuple(names)


def _waves_x2(waves: Any) -> int:
    """Twice a wave bound of a select cell, an int: the bounds are multiples of 1/2 (a program of two CTAs per head
    runs k x SM heads on 2 k CTAs per SM), so a half-wave edge compares exactly."""

    doubled = 2 * waves
    if isinstance(waves, bool) or doubled != int(doubled) or doubled <= 0:
        raise ValueError(
            f"a Cake GDP decode wave bound must be a positive multiple of 1/2, got {waves!r}"
        )
    return int(doubled)


def _selects(
    rule: Any,
    state_heads: int,
    sm_count: int,
    arch: str,
    *,
    philox_rounds: int = 0,
    snapshot: bool = False,
) -> bool:
    """Whether a program's rule admits ``state_heads = B x HV`` on a device of ``sm_count`` SMs of ``arch`` in the
    launch class (``philox_rounds``, ``snapshot``): ``above < state_heads <= at_most`` (``at_most`` ``None`` =
    unbounded) within the wave bounds ``state_heads <= ctas_per_sm_at_most x sm_count`` / ``state_heads >
    ctas_per_sm_above x sm_count`` and the inclusive ``state_heads >= ctas_per_sm_at_least x sm_count`` /
    ``state_heads < ctas_per_sm_below x sm_count`` when the rule carries them.  An FP16 record's cell may also bound
    the launch class: ``rounds_at_least <= philox_rounds <= rounds_at_most`` (inclusive; an absent bound is open;
    ``philox_rounds`` is 0 for a call that rounds to nearest even) and ``snapshot`` equal to the cell's when it
    names one (``True`` for a call that stores per-token snapshots).  A cell naming ``arches`` applies on those
    architectures only (a band that differs per architecture); a list of such cells is a band of several intervals
    and admits when any cell does.  ``None`` (a runtime program without cells) admits everything.  Each wave bound
    is a positive multiple of 1/2 and compares exactly."""

    if rule is None:
        return True
    if isinstance(rule, list):
        return any(
            _selects(
                cell,
                state_heads,
                sm_count,
                arch,
                philox_rounds=philox_rounds,
                snapshot=snapshot,
            )
            for cell in rule
        )
    kind = rule["kind"]
    if kind != "state_heads_between":
        raise ValueError(f"unknown Cake GDP decode selection rule: {kind!r}")
    arches = rule.get("arches")
    if arches is not None and arch not in arches:
        return False
    rounds_at_least, rounds_at_most = (
        rule.get("rounds_at_least"),
        rule.get("rounds_at_most"),
    )
    if rounds_at_least is not None and philox_rounds < int(rounds_at_least):
        return False
    if rounds_at_most is not None and philox_rounds > int(rounds_at_most):
        return False
    wanted = rule.get("snapshot")
    if wanted is not None and bool(snapshot) != bool(wanted):
        return False
    state_heads, sm_count = int(state_heads), int(sm_count)
    at_most = rule["at_most"]
    if state_heads <= int(rule["above"]) or (
        at_most is not None and state_heads > int(at_most)
    ):
        return False
    waves_at_most, waves_above, waves_at_least, waves_below = (
        rule.get("ctas_per_sm_at_most"),
        rule.get("ctas_per_sm_above"),
        rule.get("ctas_per_sm_at_least"),
        rule.get("ctas_per_sm_below"),
    )
    state_heads2 = 2 * state_heads
    if waves_at_most is not None and state_heads2 > _waves_x2(waves_at_most) * sm_count:
        return False
    if waves_above is not None and state_heads2 <= _waves_x2(waves_above) * sm_count:
        return False
    if (
        waves_at_least is not None
        and state_heads2 < _waves_x2(waves_at_least) * sm_count
    ):
        return False
    if waves_below is not None and state_heads2 >= _waves_x2(waves_below) * sm_count:
        return False
    return True


def selected_module(
    arch: str,
    num_householder: int,
    num_tokens: int,
    state_heads: int,
    *,
    sm_count: int,
    use_qk_l2norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
    philox_rounds: int = 0,
    use_ssm_state_indices: bool = False,
) -> str:
    """The one module a public call launches on ``arch`` for ``num_tokens`` tokens of ``state_heads = B x HV``
    state heads on a device of ``sm_count`` SMs (``sm_count(device_index)``) over a ``state_dtype`` state pool, in
    the launch class of the call: its Philox round count ``philox_rounds`` (0 for a call that rounds to nearest
    even and for an FP32 pool) and whether it stores per-token snapshots (``use_ssm_state_indices``).  Both are
    host values, so the choice reads no tensor.

    This is the band rule of the kernel (its ``band_variant``), spelled once as the records' ``select_rule``
    leaves: a route carries static-token-count programs (``build["t_steps"]`` = the call's ``T``), each owning a
    band of state-head counts (``above < state_heads <= at_most`` within optional wave bounds, multiples of half the
    SM count; a band of several intervals is a list of such cells, and a cell may name the ``arches`` it applies to),
    and runtime-token-count programs (``build["t_steps"]`` ``None``, ``T`` up to ``build["max_t_steps"]``) for
    every other call.  Exactly one static program may admit a call; else the admitting runtime program with the
    fewest ``max_t_steps`` runs (two such programs are an error).  Only the records of the pool's dtype take part
    (``build["state_dtype"]``, ``"f32"`` when absent): an FP16 record has the band of its FP32 twin less the cells
    of the row-split programs (SM90's FP16-only ``l8r2h`` programs, which take a call of their cells instead), so
    outside those cells both pools select the same program body and variant.  The cells of an FP16 record may also bound the launch class
    (:func:`_selects`): a ``snapshot_noise`` sibling admits the stochastic calls with snapshots of its cell and its
    plain program every other call, so they split the band of one FP32 program between them.  An FP32 record has
    no class keys, so its choice does not depend on the launch class.

    The choice is a function of these host values only, so it is cached: an eager call pays one lookup.
    """

    return _selected_module_cached(
        arch,
        int(num_householder),
        int(num_tokens),
        int(state_heads),
        int(sm_count),
        bool(use_qk_l2norm),
        state_dtype,
        int(philox_rounds),
        bool(use_ssm_state_indices),
    )


@functools.lru_cache(maxsize=4096)
def _selected_module_cached(
    arch: str,
    num_householder: int,
    num_tokens: int,
    state_heads: int,
    sm_count: int,
    use_qk_l2norm: bool,
    state_dtype: torch.dtype,
    philox_rounds: int,
    snapshot: bool,
) -> str:
    """:func:`selected_module` of normalized host values (an error is raised again on every call, not cached)."""

    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if sm_count < 1:
        raise ValueError("sm_count must be positive")
    route_modules(arch, num_householder, use_qk_l2norm)  # an unknown route raises here
    names = _route_names(arch, num_householder, use_qk_l2norm, state_dtype)
    if not names:
        raise ValueError(
            f"the {route_key(num_householder, use_qk_l2norm)!r} route on {arch!r} has no program for a "
            f"{state_dtype} state pool"
        )

    def admits(name: str) -> bool:
        return _selects(
            MODULES[name]["select_rule"],
            state_heads,
            sm_count,
            arch,
            philox_rounds=philox_rounds,
            snapshot=snapshot,
        )

    static = [
        name
        for name in names
        if MODULES[name]["build"]["t_steps"] == num_tokens and admits(name)
    ]
    if len(static) > 1:
        raise ValueError(
            f"{len(static)} static programs of the {route_key(num_householder, use_qk_l2norm)!r} route on {arch!r} "
            f"admit T={num_tokens}, {state_heads} state heads, philox_rounds={philox_rounds}, "
            f"use_ssm_state_indices={snapshot}: {static}"
        )
    if static:
        return static[0]
    runtime = [
        name
        for name in names
        if MODULES[name]["build"]["t_steps"] is None
        and MODULES[name]["build"]["max_t_steps"] >= num_tokens
        and admits(name)
    ]
    if not runtime:
        raise ValueError(
            f"no exported module of the {route_key(num_householder, use_qk_l2norm)!r} route on {arch!r} admits "
            f"T={num_tokens} tokens, {state_heads} state heads, philox_rounds={philox_rounds}, "
            f"use_ssm_state_indices={snapshot} (T at most "
            f"{max_num_tokens(arch, num_householder, use_qk_l2norm, state_dtype=state_dtype)})"
        )
    fewest = min(MODULES[name]["build"]["max_t_steps"] for name in runtime)
    tied = [name for name in runtime if MODULES[name]["build"]["max_t_steps"] == fewest]
    if len(tied) > 1:
        raise ValueError(
            f"{len(tied)} runtime programs of the {route_key(num_householder, use_qk_l2norm)!r} route on {arch!r} "
            f"admit T={num_tokens}, {state_heads} state heads, philox_rounds={philox_rounds}, "
            f"use_ssm_state_indices={snapshot}: {tied}"
        )
    return tied[0]


# Prefill: the single-pass route, the six flags a ``prefill_flags`` select rule
# spells and the logical argument names the prefill records map their kernel
# parameters to.
PREFILL_ROUTE = "prefill"
PREFILL_FLAGS = (
    "use_initial_state",
    "use_state_indices",
    "store_final_state",
    "unit_gates",
    "unit_beta",
    "segment_pass",
)
PREFILL_LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "o",
    "gate",
    "beta",
    "cu_seqlens",
    "state_indices",
    "output_slot_indices",
    "initial_state",
    "output_state",
    "tensormap_workspace",
    "ainv_workspace",
    "items",
    "split_workspace",
    "output_state_indices",
    "rand_seed",
)
PREFILL_LOGICAL_SCALARS = (
    "initial_state_stride_slot",
    "output_state_stride_slot",
    "scale",
    "num_seqs",
    "total_tokens",
    "num_heads",
    "num_householder",
    "total_tiles",
    "metadata_i64_mask",
    "ainv_chunk_stride",
    "state_stride_slot",
    "num_items",
    "flags",
    "philox_rounds",
)


def prefill_route_modules(arch: str, route: str = PREFILL_ROUTE) -> tuple[str, ...]:
    """Ordered module names of one prefill route on ``arch`` (``prefill``: the single-pass programs)."""

    names = ROUTES.get(arch, {}).get(route)
    if not names:
        raise ValueError(
            f"unsupported Cake GDP prefill route {route!r} on {arch!r}; exported routes on this architecture: "
            f"{sorted(ROUTES.get(arch, {}))}"
        )
    return tuple(names)


def prefill_route_applies(
    *, device_capability: Sequence[int], num_householder: int
) -> bool:
    """Whether the verified export owns one prefill call (the single-pass route exists on the device; on the
    Blackwell body, a program built for ``num_householder``)."""

    arch = arch_for_capability(device_capability)
    if arch is None or int(num_householder) not in NUM_HOUSEHOLDER:
        return False
    if arch in V6_ARCHES:
        return int(num_householder) in v6_householders(arch)
    return bool(ROUTES.get(arch, {}).get(PREFILL_ROUTE))


def _prefill_flags(rule: Any) -> dict[str, bool]:
    """The flags a ``prefill_flags`` select rule spells: the six of ``PREFILL_FLAGS`` and any further flag of the
    record's build signature the rule names."""

    if not isinstance(rule, dict) or rule.get("kind") != "prefill_flags":
        raise ValueError(f"not a Cake GDP prefill selection rule: {rule!r}")
    missing = [flag for flag in PREFILL_FLAGS if flag not in rule]
    if missing:
        raise ValueError(
            f"the Cake GDP prefill selection rule {rule!r} lacks the flags {missing}"
        )
    return {flag: bool(value) for flag, value in rule.items() if flag != "kind"}


def selected_prefill_module(
    arch: str,
    *,
    body: str | None = None,
    use_initial_state: bool,
    use_state_indices: bool,
    store_final_state: bool,
    unit_gates: bool = False,
    unit_beta: bool = False,
    state_dtype: str = "f32",
    **flags: bool,
) -> str:
    """The one single-pass program of the ``prefill`` route on ``arch`` whose flags are the call's: the initial
    state read (``use_initial_state``), the state pools addressed through slot indices (``use_state_indices``),
    the final state stored (``store_final_state``) and the gates / betas given as tensors (``unit_gates`` /
    ``unit_beta`` off) or absent (``g=None`` / ``beta=None``: the unit programs; both absent is no shipped
    program); never a segment pass.  ``flags`` names further flags of the records' build signatures: a rule not
    spelling one of them is read as off, and a program whose rule spells a flag on is selected only when it is
    asked for.  ``body`` restricts the search to the programs of one kernel body (``prefill_sm90_v6``, the V6 body,
    or ``prefill_sm90``, the merged body: ``prefill_body_class`` of the BODY ROUTE OF RECORD below picks the body of
    every sequence); None searches every body of the route.  ``state_dtype`` is the state dtype the program's build
    stores (``"f32"``, the default, for a build that spells none; the V6 pool programs with the write slot table,
    ``use_output_state_indices``, also store ``"bf16"`` and ``"f16"``).  Exactly one program must match."""

    wanted = {
        "use_initial_state": bool(use_initial_state),
        "use_state_indices": bool(use_state_indices),
        "store_final_state": bool(store_final_state),
        "unit_gates": bool(unit_gates),
        "unit_beta": bool(unit_beta),
        "segment_pass": False,
        **{flag: bool(value) for flag, value in flags.items()},
    }
    matches = []
    for name in prefill_route_modules(arch, PREFILL_ROUTE):
        if body is not None and MODULES[name]["body"] != body:
            continue
        if MODULES[name]["select_rule"].get("kind") != "prefill_flags":
            continue  # the Blackwell body's programs: ``selected_v6_module``
        if str(MODULES[name]["build"].get("state_dtype", "f32")) != state_dtype:
            continue
        spelled = _prefill_flags(MODULES[name]["select_rule"])
        if all(
            spelled.get(flag, False) == value for flag, value in wanted.items()
        ) and not any(value for flag, value in spelled.items() if flag not in wanted):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"{len(matches)} programs of the {PREFILL_ROUTE!r} route on {arch!r} of body {body!r} with a "
            f"{state_dtype} state spell the flags {wanted}: {matches}"
        )
    return matches[0]


# The gate-saturation split of the Hopper prefill body: the
# ``split_items`` programs run a long sequence as ``p`` independent items per
# head over the host's item table (formal 9, ``output_slot_indices``: four int32
# words per item, ``total_tiles`` = items).  The rules below are the rules of
# record the kernel is built and tested against: one text in every copy.
PREFILL_SPLIT_FLAG = "split_items"

# ---- SPLIT RULES OF RECORD (GDP prefill gate-saturation split, arch-keyed; one text in every copy) ----
SPLIT_CHUNK_ROWS = 64  # expanded rows per chunk; chunks come in pairs of 128 rows
SPLIT_MIN_CHUNKS = 24  # a sequence of fewer chunks is one item
SPLIT_MAX = 48  # at most 48 cuts per (sequence, head); the device's forward scan runs at most this many steps
SPLIT_WARMUP_MAX = 8  # a cut's warm-up runs at most 8 chunks from a zero seed
# The log2 gate decay at or below which a cut's warm-up saturates it, per arch: sm_90a -24 (2^-24); Blackwell (sm_100a /
# sm_103a) -48, a subset of the 2^-24 cuts (the V6 programs' BF16 state images need the deeper decay to keep
# the split on / off within 1e-6 rel-L2; -32 did not)
SPLIT_LOG2_THRESHOLD_ARCH = {"sm_90a": -24.0, "sm_100a": -48.0, "sm_103a": -48.0}
# A cut's warm-up is the fewest chunks before it whose log2 gate decay reaches the threshold, within the build's warm-up
# cap (SPLIT_WARMUP_MAX, or an inference build's longer cap) and the sequence start; a cut with none is not saturated.
# A later cut of a saturated item ends its window iff the decay total of its whole warm-up window reaches the threshold
# (gates lie in (0, 1], so the total reaches it iff a warm-up does). An inference build sums each token's log2 decay in
# integer units of 2^-16, rounded to nearest: a window's total, and so its cut's decision, is the same in every order.
# (H above, least span in chunks) per arch: sm_90a H > 24 -> 48, H > 12 -> 24, else 12; Blackwell (sm_100a / sm_103a)
# H > 12 -> 48, else 12 (its chunk step is shorter, so an item needs more chunks to pay its fill and drain)
SPLIT_SPAN_MIN_ARCH = {
    "sm_90a": ((24, 48), (12, 24), (0, 12)),
    "sm_100a": ((24, 48), (12, 48), (0, 12)),
    "sm_103a": ((24, 48), (12, 48), (0, 12)),
}
SPLIT_COST_CHUNK = 12  # the cost model, in tenths of a chunk period: per chunk of an item (span + 1 chunks) ...
SPLIT_COST_ITEM = 40  # ... and per item (seed, pipeline fill and drain, final store)
SPLIT_MODES = ("auto", "exact")
# V8-BAL: the balanced cut of a long Blackwell sequence. Its head-major chunk space (H heads of
# ``split_align``-padded chunk counts) is cut into SPLIT_BAL_SLOTS[arch] slots of equal aligned length (the arch's SM
# count, a constant: never the batch or the device); a slot crossing a head boundary is two items. A sequence takes it
# iff its slot is at least SPLIT_BAL_MIN_SLOT chunks and shorter than one head (so a slot touches at most two heads),
# a head holds at most SPLIT_MAX cuts (the device's forward scan reaches every cut), and its 2 S balanced rows are no
# more than the H p items of its uniform cut on S SMs (the balanced cut never raises the sequence's item count).
SPLIT_BAL_SLOTS = {"sm_100a": 148, "sm_103a": 152}
SPLIT_BAL_MIN_SLOT = 96


def split_sequence_bounds(
    raw_start: int, raw_end: int, total_tokens: int
) -> tuple[int, int]:
    """The device's reading of one ``cu_seqlens`` pair (``sequence_bounds`` of the segmented path, repeated so this
    block stands alone): the start clamped to ``[0, total_tokens]``, the end to ``[start, total_tokens]``."""
    start = min(max(int(raw_start), 0), total_tokens)
    end = min(max(int(raw_end), start), total_tokens)
    return start, end


def split_chunks(seqlen: int, n: int) -> int:
    """The chunks of a sequence as the device counts them: ``2 ceil(seqlen n / 128)`` (pairs of 64-row chunks)."""
    return 2 * (-(-(int(seqlen) * int(n)) // (2 * SPLIT_CHUNK_ROWS)))


def split_align(n: int) -> int:
    """The alignment of a cut in chunks: a multiple of 2 (chunk pairs) and of ``n`` (a token's ``n`` expanded rows
    stay in one item), ``lcm(2, n)`` for ``n`` in 1..4."""
    return n if n % 2 == 0 else 2 * n


def split_span_min(H: int, arch: str = "sm_90a") -> int:
    """The least span of an item under ``H`` heads on ``arch`` (SPLIT_SPAN_MIN_ARCH): a wide batch keeps its items
    long."""
    if arch not in SPLIT_SPAN_MIN_ARCH:
        raise ValueError(f"arch is one of {tuple(SPLIT_SPAN_MIN_ARCH)}, got {arch!r}")
    floors = SPLIT_SPAN_MIN_ARCH[arch]
    for h_above, span_min in floors:
        if H > h_above:  # noqa: SIM300
            return span_min
    return floors[-1][1]


def split_log2_threshold(arch: str = "sm_90a") -> float:
    """The saturation threshold of ``arch`` (SPLIT_LOG2_THRESHOLD_ARCH): the device's warm-up scan and the host model
    decide every cut against it."""
    if arch not in SPLIT_LOG2_THRESHOLD_ARCH:
        raise ValueError(
            f"arch is one of {tuple(SPLIT_LOG2_THRESHOLD_ARCH)}, got {arch!r}"
        )
    return SPLIT_LOG2_THRESHOLD_ARCH[arch]


def split_span(chunks: int, p: int, n: int) -> int:
    """The span of ``p`` items over ``chunks`` chunks: ``ceil(chunks / p)`` rounded up to the alignment."""
    align = split_align(n)
    return align * (-(-(-(-chunks // p)) // align))  # noqa: B002


def split_count(
    chunks: int,
    H: int,
    n: int,
    sm_count: int,
    split_mode: str = "auto",
    arch: str = "sm_90a",
) -> tuple[int, int]:
    """``(p, span)`` of a sequence of ``chunks`` chunks under ``H`` heads with ``n`` householder steps on ``sm_count``
    SMs of ``arch``: the item count and the cut spacing in chunks (cuts at ``j span``, ``j = 1 .. p - 1``; the last
    item is shorter when ``p span > chunks``). ``(1, 0)`` for an empty sequence; ``(1, chunks)`` for ``"exact"`` or
    below SPLIT_MIN_CHUNKS; else, over ``p = 1 .. SPLIT_MAX`` while the span is at least ``split_span_min(H, arch)``,
    the ``p`` of the least cost ``ceil(H p_eff / sm_count) ((span + 1) SPLIT_COST_CHUNK + SPLIT_COST_ITEM)`` (the
    waves of one sequence's ``H p_eff`` items of ``span + 1`` chunks each, with a per-item cost; ``p_eff = ceil(chunks
    / span)``), ties to the smaller ``p``. The rule reads the sequence's own chunk count and the shape constants
    only, never the batch: a sequence splits the same alone and in any batch."""
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode is one of {SPLIT_MODES}, got {split_mode!r}")
    if chunks <= 0:
        return 1, 0
    if split_mode == "exact" or chunks < SPLIT_MIN_CHUNKS:
        return 1, chunks
    span_min = split_span_min(H, arch)
    best = None
    for p in range(1, SPLIT_MAX + 1):
        span = chunks if p == 1 else split_span(chunks, p, n)
        if p > 1 and span < span_min:
            break  # spans only shrink with p
        p_eff = -(-chunks // span)
        waves = -(-(H * p_eff) // sm_count)
        cost = waves * ((span + 1) * SPLIT_COST_CHUNK + SPLIT_COST_ITEM)
        if best is None or cost < best[0]:
            best = (cost, p_eff, span)
    return best[1], best[2]


def split_table(
    cu_seqlens,
    n: int,
    H: int,
    sm_count: int,
    split_mode: str = "auto",
    total_tokens: int | None = None,
    arch: str = "sm_90a",
) -> list[tuple[int, int, int, int]]:
    """The work items of a batch, the rows of the device's item table (formal 9 of the ``_split`` programs, four int32
    words each; ``total_tiles`` is the row count): per sequence in order, j-major (``j = 0`` of every head, then
    ``j = 1``, ...), ``(seq, head, cut = j span, span)`` for ``(p, span) = split_count(...)`` of the sequence; an empty
    sequence has one item per head (``cut 0, span 0``: the copy of its initial state). ``cu_seqlens`` is any sequence
    of ``N + 1`` integers; ``total_tokens`` defaults to its last entry."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    items = []
    for seq in range(len(raw) - 1):
        start, end = split_sequence_bounds(raw[seq], raw[seq + 1], total)
        chunks = split_chunks(end - start, n)
        slot = split_bal_slot(chunks, H, n, split_mode, arch)
        if slot:
            items.extend(split_bal_rows(seq, chunks, H, n, slot, arch))
            continue
        p, span = split_count(chunks, H, n, sm_count, split_mode, arch)
        items.extend((seq, head, j * span, span) for j in range(p) for head in range(H))
    return items


def split_bal_slot(
    chunks: int, H: int, n: int, split_mode: str = "auto", arch: str = "sm_90a"
) -> int:
    """The V8-BAL slot of a sequence of ``chunks`` chunks under ``H`` heads on ``arch``: ``Q ceil(ceil(H C' / S) / Q)``
    with ``Q = split_align(n)``, ``C' = Q ceil(chunks / Q)`` (the head's aligned length) and ``S =
    SPLIT_BAL_SLOTS[arch]``; 0 (the uniform cut of ``split_count``) for ``"exact"``, an arch without slots, fewer than
    SPLIT_MIN_CHUNKS chunks, a slot below SPLIT_BAL_MIN_SLOT or not shorter than ``C'``, more than SPLIT_MAX cuts in
    a head (``C' > (SPLIT_MAX - 1) slot``), or more balanced rows ``2 S`` than the ``H p`` items of the uniform cut
    ``split_count(chunks, H, n, S)``."""
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode is one of {SPLIT_MODES}, got {split_mode!r}")
    if (
        split_mode == "exact"
        or arch not in SPLIT_BAL_SLOTS
        or chunks < SPLIT_MIN_CHUNKS
    ):
        return 0
    align = split_align(n)
    padded = align * (-(-chunks // align))
    slot = align * (-(-(-(-(H * padded) // SPLIT_BAL_SLOTS[arch])) // align))  # noqa: B002
    if not SPLIT_BAL_MIN_SLOT <= slot < padded <= (SPLIT_MAX - 1) * slot:
        return 0
    p, _span = split_count(chunks, H, n, SPLIT_BAL_SLOTS[arch], split_mode, arch)
    return slot if 2 * SPLIT_BAL_SLOTS[arch] <= H * p else 0


def split_bal_rows(
    seq: int, chunks: int, H: int, n: int, slot: int, arch: str
) -> list[tuple[int, int, int, int]]:
    """The item rows of a V8-BAL sequence: ``2 S`` rows, ``S = SPLIT_BAL_SLOTS[arch]``. Slot ``k`` covers the
    head-major chunks ``[k slot, (k + 1) slot)`` of ``H`` heads of ``C'`` chunks; its pieces are ``(seq, head, cut,
    -slot)`` rows (a negative span marks the balanced grid: the device steps a forward scan to the next slot boundary
    of the head-major space, ``slot`` apart), its piece that starts a head (cut 0) in row ``k`` and its other piece in
    row ``S + k``, so every CTA runs its seed items before the owners that wait on them; an absent piece is a pad row
    ``(seq, head, chunks, -slot)`` (no chunk, no store)."""
    align = split_align(n)
    padded = align * (-(-chunks // align))
    slots = SPLIT_BAL_SLOTS[arch]
    first, second = [], []
    for k in range(slots):
        lo, hi = k * slot, min((k + 1) * slot, H * padded)
        pieces = []
        while lo < hi:
            head, cut = lo // padded, lo % padded
            end = min(hi, (head + 1) * padded)
            if cut < chunks:
                pieces.append((seq, head, cut, -slot))
            lo = end
        pad = (seq, min(k * slot // padded, H - 1), chunks, -slot)
        if len(pieces) == 2 and pieces[1][2] == 0:
            pieces.reverse()
        first.append(pieces[0] if pieces else pad)
        second.append(pieces[1] if len(pieces) > 1 else pad)
    return first + second


# ---- END SPLIT RULES OF RECORD ----

# ---- BODY ROUTE OF RECORD (GDP prefill sm_90a body choice per sequence; one text in every copy) ----
ROUTE_BODY_P3 = "prefill_sm90"  # the merged body: every contract flag, the split items, the unit programs
ROUTE_BODY_V6 = "prefill_sm90_v6"  # the V6 fork-order body: h0 seed, slot pool, one item per (seq, head)
ROUTE_BODIES = (ROUTE_BODY_P3, ROUTE_BODY_V6)
ROUTE_STEP_COST_TENTHS = 36  # a merged-body chunk-step costs 3.6 V6 steps: 7.926 us per merged-body step over 2.203 us per V6 step, both with an initial state, paired H100 measurements over 18 rows (2026-10-27); the fit from the zero-seed V6 step was 31
# every split item is costed with its warm-up from a zero seed
ROUTE_WARMUP_CHUNKS = SPLIT_WARMUP_MAX
ROUTE_MIN_CHUNKS_P3 = 384  # below it the split's warm-up does not pay: every 18-row case under 384 chunks is a V6 win


def route_chunk_steps(cu_seqlens, n: int, total_tokens: int | None = None) -> list[int]:
    """The chunk count of every sequence of the batch: ``split_chunks`` of its clamped length (``split_sequence_bounds``
    of ``cu_seqlens``, any sequence of ``N + 1`` integers; ``total_tokens`` defaults to its last entry)."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    chunks = []
    for seq in range(len(raw) - 1):
        start, end = split_sequence_bounds(raw[seq], raw[seq + 1], total)
        chunks.append(split_chunks(end - start, n))
    return chunks


def prefill_body_costs(
    seq_chunks: int,
    H: int,
    n: int,
    sm_count: int,
) -> tuple[int, int]:
    """The cost of one sequence of ``seq_chunks`` chunks under each body as if it were the whole call, ``(cost_v6,
    cost_p3)`` in tenths of a V6 chunk-step: its chunk-steps over the CTAs its grid occupies (``min(sm_count, items)``),
    a V6 step costing 10 and a merged-body step ``ROUTE_STEP_COST_TENTHS``. The V6 body runs one item per head over the
    sequence's chunks; the merged body runs the ``split_count`` items per head, each over ``ceil(chunks / p)`` chunks
    and the warm-up of ``ROUTE_WARMUP_CHUNKS``. An empty sequence is ``(0, 0)``."""
    if seq_chunks <= 0 or H <= 0:
        return 0, 0
    p, _ = split_count(seq_chunks, H, n, sm_count)
    cost_v6 = 10 * -(-(H * seq_chunks) // min(sm_count, H))
    items = H * p
    item_steps = -(-seq_chunks // p) + ROUTE_WARMUP_CHUNKS
    waves_p3 = -(-(items * item_steps) // min(sm_count, items))
    cost_p3 = ROUTE_STEP_COST_TENTHS * waves_p3
    return cost_v6, cost_p3


def prefill_body_class(
    seq_chunks: int,
    H: int,
    n: int,
    sm_count: int,
    *,
    store_final_state: bool = True,
    unit_gates: bool = False,
    unit_beta: bool = False,
) -> str:
    """The body of one sequence: a function of its own chunk count, ``H``, ``n``, the SM count and the call's flags,
    never of the batch (a sequence runs the same body alone and in any batch). ``ROUTE_BODY_P3`` for a unit-gate /
    unit-beta call (no V6 unit program); ``ROUTE_BODY_V6`` for a call without a final state (no merged split program
    there, and the V6 step is the cheaper), for a sequence under ``ROUTE_MIN_CHUNKS_P3`` chunks (an empty one included:
    both bodies copy its initial state bitwise) and whenever its merged cost is not below its V6 cost; ``ROUTE_BODY_P3``
    otherwise -- ``split_count`` then cuts such a sequence (``p >= 2``), so the merged class always runs a split
    program."""
    if unit_gates or unit_beta:
        return ROUTE_BODY_P3
    if not store_final_state or seq_chunks < ROUTE_MIN_CHUNKS_P3:
        return ROUTE_BODY_V6
    cost_v6, cost_p3 = prefill_body_costs(seq_chunks, H, n, sm_count)
    return ROUTE_BODY_P3 if cost_p3 < cost_v6 else ROUTE_BODY_V6


def prefill_body_classes(
    cu_seqlens,
    n: int,
    H: int,
    sm_count: int,
    *,
    store_final_state: bool = True,
    unit_gates: bool = False,
    unit_beta: bool = False,
    total_tokens: int | None = None,
) -> list[str]:
    """The body of every sequence of a batch: ``prefill_body_class`` over ``route_chunk_steps``."""
    return [
        prefill_body_class(
            chunks,
            H,
            n,
            sm_count,
            store_final_state=store_final_state,
            unit_gates=unit_gates,
            unit_beta=unit_beta,
        )
        for chunks in route_chunk_steps(cu_seqlens, n, total_tokens)
    ]


def prefill_launch_plan(classes) -> list[tuple[str, list[int]]]:
    """The launches of a batch from its sequences' bodies: one per body present, in ``ROUTE_BODIES`` order, each with
    the ascending indices of the sequences of that class. One body -> one launch of the whole batch (every performance
    row); two -> two launches, each over its own sequences' tokens, state slots and O rows alone (no row is touched by
    both), run on the indexed program of its body with the dense state tensors as pools and ``slot = sequence``."""
    plan = []
    for body in ROUTE_BODIES:
        seqs = [seq for seq, cls in enumerate(classes) if cls == body]
        if seqs:
            plan.append((body, seqs))
    return plan


# ---- END BODY ROUTE OF RECORD ----

# ---- BATCH BODY ROUTE (GDP prefill sm_90a: one body, one launch per call) ----
# The backend runs a whole sm_90a call on ONE body in ONE launch; the per-sequence classes of the route of record only
# decide whether the merged split body is a candidate (some sequence is merged-class).  The body is then the one of the
# lesser modeled kernel time of the whole batch.  Both bodies run persistent CTAs that take their work strided (CTA
# ``b`` of a grid of ``G`` runs work ``b, b + G, ...``) and a CTA's time is the sum of its chunk-steps, so the model
# is the most loaded CTA: the V6 body over the ``N x H`` sequence-major tiles of each sequence's chunks, the merged
# body over the ``split_table`` items (each item's chunks, plus ``ROUTE_WARMUP_CHUNKS`` for a cut past 0) on
# ``min(sm_count, tiles)`` CTAs.  The step and base times are a fit of the H100 (132 SMs) kernel times
# (2026-11-24: 88 workloads x two gate sets, each body forced on the whole batch; error within 3 %).
ROUTE_BATCH_V6_STEP_NS = 2040  # one V6 chunk-step
ROUTE_BATCH_V6_BASE_NS = 15000  # the V6 launch, seed and final store
ROUTE_BATCH_P3_STEP_NS = 7570  # one merged-body chunk-step
ROUTE_BATCH_P3_BASE_NS = 27000  # the merged launch, seed and final store


def _most_loaded(loads: list[int], grid: int) -> int:
    """The largest sum of ``loads[b::grid]`` over the ``grid`` CTAs (0 for no work)."""
    if not loads:
        return 0
    return max(sum(loads[b::grid]) for b in range(grid))


def prefill_batch_costs(
    cu_seqlens,
    n: int,
    H: int,
    sm_count: int,
    total_tokens: int | None = None,
) -> tuple[int, int]:
    """The modeled kernel time of the whole batch under each body, ``(t_v6, t_p3)`` in nanoseconds: the base plus the
    chunk-steps of the most loaded CTA at the body's step time (see the BATCH BODY ROUTE comment).  The merged body
    runs the items of ``split_table`` (a sequence the rule leaves whole is one item per head)."""
    chunks = route_chunk_steps(cu_seqlens, n, total_tokens)
    tiles = [c for c in chunks for _ in range(H)]
    if not tiles:
        return 0, 0
    t_v6 = ROUTE_BATCH_V6_BASE_NS + ROUTE_BATCH_V6_STEP_NS * _most_loaded(
        tiles, min(sm_count, len(tiles))
    )
    items = [
        min(span, chunks[seq] - cut) + (ROUTE_WARMUP_CHUNKS if cut > 0 else 0)
        for seq, _head, cut, span in split_table(
            cu_seqlens, n, H, sm_count, total_tokens=total_tokens
        )
    ]
    t_p3 = ROUTE_BATCH_P3_BASE_NS + ROUTE_BATCH_P3_STEP_NS * _most_loaded(
        items, min(sm_count, len(items))
    )
    return t_v6, t_p3


def prefill_batch_body(
    cu_seqlens,
    n: int,
    H: int,
    sm_count: int,
    *,
    store_final_state: bool = True,
    unit_gates: bool = False,
    unit_beta: bool = False,
    total_tokens: int | None = None,
) -> str:
    """The one body of a whole sm_90a call: ``ROUTE_BODY_P3`` for a unit-gate / unit-beta call (its exact unit
    program), ``ROUTE_BODY_V6`` for a call without a final state and for a batch with no merged-class sequence (the
    route of record), else the body of the lesser ``prefill_batch_costs`` (ties to V6).  A merged choice runs the
    split program over ``split_table`` of the whole batch: every sequence, a short or empty one included, is in the
    same launch (a sequence the rule leaves whole is one item per head, bitwise the merged single pass)."""
    if unit_gates or unit_beta:
        return ROUTE_BODY_P3
    if not store_final_state:
        return ROUTE_BODY_V6
    classes = prefill_body_classes(
        cu_seqlens, n, H, sm_count, store_final_state=True, total_tokens=total_tokens
    )
    if ROUTE_BODY_P3 not in classes:
        return ROUTE_BODY_V6
    t_v6, t_p3 = prefill_batch_costs(cu_seqlens, n, H, sm_count, total_tokens)
    return ROUTE_BODY_P3 if t_p3 < t_v6 else ROUTE_BODY_V6


# ---- END BATCH BODY ROUTE ----


# ---- SM90 V6 WARM-UP SPLIT PLAN (GDP prefill sm_90a: the split twins of the V6 programs) ----
# The V6 body's ``split_items`` programs (the plain, initial-state and pool-indexed flag sets, with per-CTA row lists
# and the stale-plan guard) run one launch of persistent CTAs, one per SM, over a host table of rows ``(sequence,
# head, cut, span)`` in chunk units (a chunk is 64 expanded rows: ``ceil(len n / 64)`` chunks per sequence):
# * A sequence of ``chunks`` chunks is balanced when its slot ``slot = n ceil(ceil(H padded / G) / n)`` (``padded =
#   n ceil(chunks / n)``: its head-major chunk space cut into ``G`` = SM count slots, each a whole number of
#   blocks) is at least V6_SM90_MIN_SLOT chunks and shorter than a head, and a head holds at most V6_SM90_SPLIT_MAX
#   cuts.  Its rows are ``(seq, head, cut, -slot)``: slot ``k`` covers chunks ``[k slot, (k + 1) slot)`` of that
#   space, one row per head it touches; a cut fewer than V6_SM90_CUT_MIN chunks after its head start moves to the
#   head start.  The cuts read the sequence's own length and the shape, never the rest of the batch.  At most
#   V6_SM90_PLAN_ROWS // 2 sequences are balanced, the longest.
# * Every other sequence is one whole-head row per head, ``(seq, head, 0, chunks)``.
# * Placement: CTA ``k`` takes slot ``k`` of every balanced sequence, its cut-0 rows first; then each whole-head row,
#   longest first, goes to the least-loaded CTA (several one-token rows fill the slack of the balanced slots).
# * Choice: the facade runs the split program when a sequence is balanced and the plan's most loaded CTA (its rows'
#   chunks, V6_SM90_ROW_COST per row and V6_SM90_WARMUP_COST per row after a cut) is lighter than the most loaded
#   CTA of the V6 program over the ``N x H`` tiles (each tile's chunks and V6_SM90_ROW_COST, strided over
#   ``min(SM count, N x H)`` CTAs): both run the same chunk step, so a batch whose tiles already fill the SMs (many
#   sequences of one or two slots each) keeps the V6 program.
# The device scans back from every cut past 0 until the log2 gate decay of the chunks before it reaches the build's
# threshold (within a warm-up cap), runs that warm-up from a zero state and writes O from the cut on; a cut that does
# not saturate is not taken (the previous row runs through it), so the result does not depend on the cuts beyond
# rounding.  The row of a head's cut 0 seeds the state and raises a seed-done flag; the row that reaches the head's
# end stores the final state after that flag (every CTA is resident at once: one CTA per SM, at most SM count CTAs).
# The table is ``[4 R row words][grid + 1 offsets][N][N + 1 cu_seqlens]``; the guarded program compares the last two
# with the device offsets and, when they differ, runs no row and writes NaN to ``o[0, 0, :2]`` and, when it stores
# one, to the final state ``[0, 0, 0, 0]``.  The split workspace (``ainv_workspace``) holds the ``N x H`` seed-done
# flags, zero at launch; the kernel leaves them zero.
# Expanded rows per chunk of the V6 body.
V6_SM90_CHUNK_ROWS = 64
# A sequence of fewer chunks is one row per head.
V6_SM90_MIN_CHUNKS = 24
# Chunks: a shorter balanced slot pays more per-row fill than it gains.
V6_SM90_MIN_SLOT = 24
# A cut fewer chunks than this after its head start moves to the head start.
V6_SM90_CUT_MIN = 24
# At most this many cuts per head (the device's forward scan reaches every cut).
V6_SM90_SPLIT_MAX = 48
# Rows of a CTA's list the device decodes into its shared plan table.
V6_SM90_PLAN_ROWS = 64
# A row's modeled cost past its chunks, in chunk steps (decode, seed, fill, final store).
V6_SM90_ROW_COST = 3
# The modeled warm-up of a row after a cut (the device decides it).
V6_SM90_WARMUP_COST = 6


class V6Sm90Plan(NamedTuple):
    """One launch of a V6 split program on sm_90a: the rows in table order, the grid (the SM count when a sequence is
    balanced) and the ``grid + 1`` offsets of the CTAs' row lists (CTA ``b`` runs rows ``offsets[b] ..
    offsets[b + 1] - 1``).  ``split`` is whether the facade runs the split program for the batch: some sequence is
    balanced and the plan's most loaded CTA, ``load``, is lighter than the V6 program's, ``serial_load`` (both in
    chunk steps)."""

    rows: tuple[tuple[int, int, int, int], ...]
    offsets: tuple[int, ...]
    grid: int
    split: bool
    load: int
    serial_load: int


def v6_sm90_chunks(
    cu_seqlens: Sequence[int], n: int, total_tokens: int | None = None
) -> list[int]:
    """The V6 chunk count of each sequence as the device reads ``cu_seqlens``: ``ceil(len n / 64)``."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    out = []
    for q in range(len(raw) - 1):
        start, end = split_sequence_bounds(raw[q], raw[q + 1], total)
        out.append(-(-(end - start) * int(n) // V6_SM90_CHUNK_ROWS))
    return out


def _v6_sm90_padded(chunks: int, n: int) -> int:
    return n * -(-chunks // n)


def v6_sm90_slot(chunks: int, H: int, n: int, slots: int) -> int:
    """The balanced slot of a sequence of ``chunks`` chunks on ``slots`` slots, 0 when it is not balanced."""
    if chunks < V6_SM90_MIN_CHUNKS:
        return 0
    padded = _v6_sm90_padded(chunks, n)
    per_slot = -(-(H * padded) // slots)
    slot = n * -(-per_slot // n)
    if not V6_SM90_MIN_SLOT <= slot < padded or padded > (V6_SM90_SPLIT_MAX - 1) * slot:
        return 0
    return slot


def _v6_sm90_slot_rows(
    seq: int, chunks: int, H: int, n: int, slot: int
) -> list[list[tuple[int, int, int, int]]]:
    """The rows of a balanced sequence per slot, each slot's cut-0 rows first."""
    padded = _v6_sm90_padded(chunks, n)
    total = H * padded
    pieces: list[list[list[int]]] = []
    for k in range(-(-total // slot)):
        lo, hi = k * slot, min((k + 1) * slot, total)
        here = []
        while lo < hi:
            head, cut = lo // padded, lo % padded
            if cut < chunks:
                here.append([seq, head, cut, -slot])
            lo = (head + 1) * padded if hi > (head + 1) * padded else hi
        pieces.append(here)
    for head in range(H):
        rows = [r for slot_rows in pieces for r in slot_rows if r[1] == head]
        first = min((r[2] for r in rows if r[2] > 0), default=None)
        if first is not None and first < V6_SM90_CUT_MIN:
            for slot_rows in pieces:
                slot_rows[:] = [
                    r for r in slot_rows if not (r[1] == head and r[2] == 0)
                ]
            for r in rows:
                if r[2] == first:
                    r[2] = 0
    return [
        [
            (r[0], r[1], r[2], r[3])
            for r in sorted(slot_rows, key=lambda r: 0 if r[2] == 0 else 1)
        ]
        for slot_rows in pieces
    ]


def _v6_sm90_row_cost(row: Sequence[int], chunks: int, n: int) -> int:
    """The modeled cost of a row in chunk steps: its chunks (to the next planned cut for a balanced row), plus
    V6_SM90_ROW_COST and, after a cut, V6_SM90_WARMUP_COST."""
    _seq, head, cut, span = row
    if span < 0:
        slot, padded = -span, _v6_sm90_padded(chunks, n)
        at = head * padded + cut
        end = min((at // slot + 1) * slot - head * padded, chunks)
        if cut == 0 and end < V6_SM90_CUT_MIN:
            end = min(end + slot, chunks)
    else:
        end = chunks
    return (
        max(end - cut, 0) + V6_SM90_ROW_COST + (V6_SM90_WARMUP_COST if cut > 0 else 0)
    )


def v6_sm90_serial_load(chunks: Sequence[int], H: int, sm_count: int) -> int:
    """The most loaded CTA of the V6 program over the ``N x H`` sequence-major tiles of sequences of ``chunks``
    chunks, strided over ``min(sm_count, N x H)`` persistent CTAs, in chunk steps (a tile costs its chunks and
    V6_SM90_ROW_COST)."""
    tiles = [c + V6_SM90_ROW_COST for c in chunks for _ in range(int(H))]
    if not tiles:
        return 0
    grid = min(int(sm_count), len(tiles))
    return max(sum(tiles[b::grid]) for b in range(grid))


def prefill_v6_sm90_plan(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    total_tokens: int | None = None,
) -> V6Sm90Plan:
    """The plan of a V6 split program for the host offsets ``cu_seqlens`` (``N + 1`` integers; ``total_tokens``
    defaults to the last) on a device of ``sm_count`` SMs (see the rule above)."""
    H, n, sm_count = int(num_heads), int(num_householder), int(sm_count)
    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")
    chunks = v6_sm90_chunks(cu_seqlens, n, total_tokens)
    slot_of = {q: v6_sm90_slot(c, H, n, sm_count) for q, c in enumerate(chunks)}
    balanced = sorted(
        sorted((q for q, s in slot_of.items() if s), key=lambda q: (-chunks[q], q))[
            : V6_SM90_PLAN_ROWS // 2
        ]
    )
    chosen = set(balanced)
    whole = [
        (q, h, 0, chunks[q])
        for q in range(len(chunks))
        if q not in chosen
        for h in range(H)
    ]
    slot_rows = {
        q: _v6_sm90_slot_rows(q, chunks[q], H, n, slot_of[q]) for q in balanced
    }
    total_rows = sum(len(r) for q in balanced for r in slot_rows[q]) + len(whole)
    grid = sm_count if balanced else max(1, min(sm_count, total_rows))
    lists: list[list[tuple[int, int, int, int]]] = [[] for _ in range(grid)]
    load = [0] * grid
    for q in balanced:
        for k, rows in enumerate(slot_rows[q]):
            for row in rows:
                lists[k].append(row)
                load[k] += _v6_sm90_row_cost(row, chunks[q], n)
    for b in range(grid):
        lists[b].sort(key=lambda r: 0 if r[2] == 0 else 1)  # stable: cut-0 rows first
        if len(lists[b]) > V6_SM90_PLAN_ROWS:
            raise ValueError(
                f"CTA {b} of the sm_90a split plan has {len(lists[b])} balanced rows, more than its plan table's "
                f"{V6_SM90_PLAN_ROWS}"
            )
    # each whole-head row, longest first, to the least-loaded CTA (ties to the lower index)
    heap = [(load[b], b) for b in range(grid)]
    heapq.heapify(heap)
    for row in sorted(whole, key=lambda r: -_v6_sm90_row_cost(r, chunks[r[0]], n)):
        cost = _v6_sm90_row_cost(row, chunks[row[0]], n)
        least, b = heapq.heappop(heap)
        lists[b].append(row)
        load[b] = least + cost
        heapq.heappush(heap, (load[b], b))
    offsets = (0, *itertools.accumulate(len(x) for x in lists))
    most = max(load, default=0)
    serial = v6_sm90_serial_load(chunks, H, sm_count)
    return V6Sm90Plan(
        rows=tuple(r for x in lists for r in x),
        offsets=offsets,
        grid=grid,
        split=bool(balanced) and most < serial,
        load=most,
        serial_load=serial,
    )


def v6_sm90_item_words(plan: V6Sm90Plan, cu_seqlens: Sequence[int]) -> tuple[int, ...]:
    """The int32 words of a V6 split program's item table (formal 9, ``output_slot_indices``): the rows (4 words
    each), the ``grid + 1`` row offsets, then the planned sequence count ``N`` and the ``N + 1`` host offsets
    ``cu_seqlens`` the plan was made from (the stale-plan guard)."""
    cu = [int(c) for c in cu_seqlens]
    words = [int(w) for row in plan.rows for w in row]
    words.extend(int(o) for o in plan.offsets)
    words.extend((len(cu) - 1, *cu))
    return tuple(words)


# ---- END SM90 V6 WARM-UP SPLIT PLAN ----


# ---- BLACKWELL V6 LAUNCH PLAN (sm_100a / sm_103a prefill) ----
# The Blackwell prefill body (``prefill_bw_v6``) runs persistent CTAs over a host item table of rows ``(sequence,
# head, cut, span)``.  A call launches the plan the PLAN CHOOSER below picks for its batch: the plan of record
# (``prefill_v6_record_plan``) or a candidate of the same two builds modeled faster.  The plan of record launches the
# ``split_items`` build over the chunk-unit rows of ``split_table`` (when ``launch_split`` holds: some sequence is
# cut, or the batch holds an empty sequence, whose state copy only the split build runs), else the block build over
# one block-unit row ``(seq, head, 0, ceil(len / 64))`` per non-empty sequence and head, longest first, CTA ``b`` of a
# grid of ``G`` running rows ``b, b + G, ...``.  The split build runs per-CTA row lists (``inference_csr``: CTA ``b``
# runs rows ``offsets[b] .. offsets[b + 1] - 1``, its cut-0 rows first): the long sequences' rows stay where the
# table spreads them and each short row goes to the least-loaded CTA.  The split rules of record read each sequence
# alone, so in the plan of record short sequences added to a long one never change its cuts; a chosen candidate may
# cut a sequence differently in another batch.  The grid is ``min(SM count, rows of the table)`` and never exceeds
# the SM count: one CTA fills an SM, so
# every CTA of a launch is resident at once, which the split build's seed-done wait needs (an owner waits only on a
# cut-0 row, which every CTA runs first and which waits on nothing).  Both builds read the planned sequence count and
# offsets after the rows (``v6_item_words``) and, when they differ from the device offsets, run no item and write NaN
# to ``o[0, 0, :2]`` (one 32-bit word) and final state ``[0, 0, 0, 0]``: a CUDA graph is valid only for the batch it
# captured.  The builds are inference builds: their results agree with the exact recurrence within FlashInfer's
# tolerance and may differ in the last bits with the composition of the batch.
V6_ARCHES = ("sm_100a", "sm_103a")
V6_BODY = "prefill_bw_v6"
V6_SELECT_KIND = "v6_launch"
V6_TENSORMAP_BYTES = 128  # one CUtensorMap slot per CTA (``tmap_ws``)
V6_SPLIT_WS_PLACEHOLDER = (
    16  # ``split_ws`` bytes of a build without the split, which never reads it
)
V6_FLAG_INITIAL_STATE = 1
V6_FLAG_FINAL_STATE = 2
# The state forms of the Blackwell body's programs: the state dtype as the records' builds spell it (``state_dtype``;
# an FP32 record spells none) and whether the program addresses slot pools (``use_state_indices``).
PREFILL_STATE_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
}


# ---- INFERENCE PLACEMENT (inference split builds: host rule, no kernel change) ----
# The V6 split build runs table row it on CTA it % grid, in table order. inference_placement re-lays a split_table
# onto the grid: each CTA b gets the long sequences' rows b, b + grid, ... of the table (the V8-BAL slot pairing and
# the uniform cut's j-major rounds stay as split_table spreads them), then each item of an unsplit sequence (one row per
# head: a short or 1-token sequence, an empty sequence's copy) goes to the CTA of least modeled load, longest first, at
# the end of the CTA's list, among the CTAs whose list is shorter than the rounds the table needs (the longest list
# of long rows plus ceil(unsplit rows / grid): every pad row costs each CTA a decode). A CTA runs its cut-0 rows of
# split sequences first (the seed-done flags: an owner waits only on a cut-0 row, which waits on nothing). The lists are laid out round-major, row r grid + b = the r-th row of
# CTA b, with pad rows (seq, 0, chunks, INFERENCE_PAD_SPAN) of a non-empty sequence where a list is shorter: the
# device echoes (chunks, 0) for a pad, runs no chunk and stores nothing. Every row of the table is kept, so the
# results are bit for bit those of the split_table launch.
INFERENCE_PAD_SPAN = -1
INFERENCE_ITEM_COST = 3  # a row's modeled cost past its chunks, in chunk steps (decode, seed, state store)
INFERENCE_WARMUP_COST = 2  # the modeled warm-up of a cut > 0 (the device decides it)


def inference_row_cost(row, chunks: int, n: int) -> int:
    """The modeled cost of split_table row ``row`` of a sequence of ``chunks`` chunks, in chunk steps."""
    _seq, head, cut, span = row
    if span < 0:
        if cut >= chunks:
            return 0
        align = split_align(n)
        padded = align * (-(-chunks // align))
        at = head * padded + cut
        end = min((at // -span + 1) * -span - head * padded, chunks)
    else:
        end = min(cut + span, chunks)
    return (
        max(end - cut, 0)
        + INFERENCE_ITEM_COST
        + (INFERENCE_WARMUP_COST if cut > 0 else 0)
    )


def inference_placement(
    rows, cu_seqlens, n: int, grid: int, total_tokens: int | None = None
):
    """The placed table of the split_table ``rows`` of ``cu_seqlens`` on ``grid`` CTAs (the rule above)."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    chunks = [
        split_chunks(e - s, n)
        for s, e in (
            split_sequence_bounds(raw[q], raw[q + 1], total)
            for q in range(len(raw) - 1)
        )
    ]
    per: dict[tuple[int, int], int] = {}
    for seq, head, _cut, _span in rows:
        per[(seq, head)] = per.get((seq, head), 0) + 1
    split_seqs = {seq for (seq, _head), count in per.items() if count > 1}
    long_rows = [r for r in rows if r[0] in split_seqs]
    short_rows = [r for r in rows if r[0] not in split_seqs]
    lists: list[list[tuple[int, int, int, int]]] = [[] for _ in range(grid)]
    load = [0] * grid
    for at, row in enumerate(long_rows):
        lists[at % grid].append(row)
        load[at % grid] += inference_row_cost(row, chunks[row[0]], n)
    for b in range(grid):
        lists[b].sort(
            key=lambda r: 0 if r[2] == 0 else 1
        )  # stable: cut-0 rows first, the rounds kept otherwise
    cap = max((len(x) for x in lists), default=0) + -(-len(short_rows) // grid)
    for row in sorted(
        short_rows, key=lambda r: -inference_row_cost(r, chunks[r[0]], n)
    ):
        b = min(
            (c for c in range(grid) if len(lists[c]) < cap), key=lambda c: (load[c], c)
        )
        lists[b].append(row)
        load[b] += inference_row_cost(row, chunks[row[0]], n)
    rounds = max((len(x) for x in lists), default=0)
    pad_seq = next((q for q, c in enumerate(chunks) if c > 0), None)
    placed = []
    for r in range(rounds):
        for b in range(grid):
            if r < len(lists[b]):
                placed.append(lists[b][r])
            else:
                assert pad_seq is not None, "a pad needs a non-empty sequence"
                placed.append((pad_seq, 0, chunks[pad_seq], INFERENCE_PAD_SPAN))
    return placed


def lpt_rounds(loads, cost: int, k: int):
    """The one-at-a-time LPT of ``k`` rows of one ``cost`` over the CTAs of ``loads``: row ``i`` goes to the CTA of the
    least ``(load, index)`` and adds ``cost`` to its load.  Computed in bulk: with ``q = load // cost`` and ``r = load
    % cost`` a CTA takes a row in round ``q`` and in every later round, and a round's rows go to the CTAs active in it
    (``q`` at most the round) in ``(r, index)`` order -- each load of a round is below every load of the next one.
    Returns ``(head, phase, full, tail)``: rows ``0 .. len(head) - 1`` go to CTAs ``head``, then ``full`` rounds of
    every CTA in the order ``phase`` (row ``len(head) + f G + j`` to ``phase[j]``, ``G`` CTAs), then the remaining
    rows to CTAs ``tail``.  A zero ``cost`` puts every row on the least-loaded CTA."""
    G = len(loads)
    if k <= 0:
        return [], [], 0, []
    if cost <= 0:
        least = min(range(G), key=lambda b: (loads[b], b))
        return [least] * k, [], 0, []
    if k <= G:  # enough CTAs at the least load: one row each, lowest index first
        low = min(loads)
        idle = [b for b, load in enumerate(loads) if load == low]
        if len(idle) >= k:
            return idle[:k], [], 0, []
    q = [load // cost for load in loads]
    wake = sorted(range(G), key=q.__getitem__)
    keys: list[int] = []  # the (r, index) keys of the active CTAs, ascending
    order: list[int] = []  # their CTAs, in that order
    head: list[int] = []
    at, placed, rnd = 0, 0, q[wake[0]]
    while at < G:
        woke = []
        while at < G and q[wake[at]] <= rnd:
            b = wake[at]
            woke.append((loads[b] - q[b] * cost) * G + b)
            at += 1
        if woke:
            keys = sorted(keys + woke) if keys else sorted(woke)
            order = [x % G for x in keys]
        if at == G:
            break
        rounds = min(q[wake[at]] - rnd, (k - placed) // len(order))
        head.extend(order * rounds)
        placed += rounds * len(order)
        rnd += rounds
        if rnd < q[wake[at]]:  # the last rows end within this round
            head.extend(order[: k - placed])
            return head, [], 0, []
    full, rest = divmod(k - placed, G)
    return head, order, full, order[:rest]


def lpt_counts(loads, cost: int, k: int) -> list[int]:
    """The rows per CTA of :func:`lpt_rounds`, in ``O(G log G)``: ``t``, the last round the rows reach, is the largest
    with ``sum(max(0, t - q))`` (the rows of the rounds before it) at most ``k``; the rest of the rows go to the CTAs
    active in round ``t`` in ``(r, index)`` order."""
    G = len(loads)
    if k <= 0:
        return [0] * G
    counts = [0] * G
    if cost <= 0:
        counts[min(range(G), key=lambda b: (loads[b], b))] = k
        return counts
    if k <= G:  # enough CTAs at the least load: one row each, lowest index first
        low = min(loads)
        idle = [b for b, load in enumerate(loads) if load == low]
        if len(idle) >= k:
            for b in idle[:k]:
                counts[b] = 1
            return counts
    q = [load // cost for load in loads]
    ordered = sorted(q)
    total = 0
    for i in range(1, G + 1):
        total += ordered[i - 1]
        t = (k + total) // i
        if i == G or t < ordered[i]:
            break
    counts = [t - x if x < t else 0 for x in q]
    rest = k - sum(counts)
    if rest:
        for x in sorted(
            (loads[b] - q[b] * cost) * G + b for b in range(G) if q[b] <= t
        )[:rest]:
            counts[x % G] += 1
    return counts


def lpt_place(loads, cost: int, rows, lists) -> None:
    """Append ``rows`` (in order) to the CTA ``lists`` by :func:`lpt_rounds` and add their cost to ``loads``."""
    G = len(loads)
    head, phase, full, tail = lpt_rounds(loads, cost, len(rows))
    for row, b in zip(rows, head, strict=False):
        lists[b].append(row)
        loads[b] += cost
    at = len(head)
    if full:
        end = at + full * G
        for j, b in enumerate(phase):
            lists[b].extend(rows[at + j : end : G])
            loads[b] += full * cost
        at = end
    for row, b in zip(rows[at:], tail, strict=False):
        lists[b].append(row)
        loads[b] += cost


def inference_lists(long_rows, long_costs, short_groups, grid: int, flags=None):
    """The per-CTA row lists of a split launch on ``grid`` CTAs: the long sequences' rows (``long_rows`` in table
    order, modeled costs ``long_costs``) row ``at`` to CTA ``at % grid``, each CTA's cut-0 rows first (``flags``: 1
    for a row of cut > 0, by default from the rows); then each ``(cost, rows)`` group of ``short_groups`` (one row per
    (sequence, head) of the unsplit sequences, groups in placement order, rows in table order), each row on the CTA
    of least ``(modeled load, index)`` (:func:`lpt_place`)."""
    count = len(long_rows)
    if count <= grid:  # at most one long row per CTA
        lists = [[row] for row in long_rows]
        lists += [[] for _ in range(grid - count)]
        loads = list(long_costs) + [0] * (grid - count)
    else:
        full, rest = divmod(count, grid)
        bounds = range(0, full * grid, grid)
        lists = [
            list(cta)
            for cta in zip(*(long_rows[at : at + grid] for at in bounds), strict=False)
        ]
        for b in range(rest):
            lists[b].append(long_rows[full * grid + b])
        loads = _v6_rr_loads(long_costs, grid)
        if flags is None:
            flags = [1 if row[2] else 0 for row in long_rows]
        # the CTAs holding a cut-0 row after a row of cut > 0: their rows are reordered (stable: cut-0 rows first)
        seen = fix = 0
        for at in range(0, count, grid):
            part = flags[at : at + grid]
            ones = int("".join(map(str, part))[::-1], 2)
            fix |= seen & ~ones & ((1 << len(part)) - 1)
            seen |= ones
        while fix:
            b = (fix & -fix).bit_length() - 1
            fix &= fix - 1
            cta = lists[b]
            lists[b] = [row for row in cta if not row[2]] + [
                row for row in cta if row[2]
            ]
    for cost, rows in short_groups:
        lpt_place(loads, cost, rows, lists)
    return lists


def inference_csr(rows, cu_seqlens, n: int, grid: int, total_tokens: int | None = None):
    """ABI v2.2 (the ``csr`` build): the split_table ``rows`` of ``cu_seqlens`` as per-CTA lists on ``grid`` CTAs,
    returned as the CTA-major table and its ``grid + 1`` offsets (CTA b runs rows ``offsets[b] .. offsets[b + 1] - 1``,
    no pad rows). The long sequences' rows go as in :func:`inference_placement` (row ``at`` of them to CTA ``at %
    grid``, each CTA's cut-0 rows first); then each unsplit sequence's row, longest first (table order among equal
    costs), to the least-loaded CTA, lowest index first (several 1-token rows stack on an idle CTA): the CTA a rule
    of "the least-loaded CTA among those whose load plus the row's cost stays within the current maximum, else the
    least-loaded CTA of all" picks too, since the least-loaded CTA has room whenever any CTA has.  Rows of one cost
    are placed in bulk (:func:`lpt_place`): ``O(R log G)``."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    chunks = [
        split_chunks(e - s, n)
        for s, e in (
            split_sequence_bounds(raw[q], raw[q + 1], total)
            for q in range(len(raw) - 1)
        )
    ]
    per = collections.Counter(map(operator.itemgetter(0, 1), rows))
    split_seqs = {seq for (seq, _head), count in per.items() if count > 1}
    long_rows = [r for r in rows if r[0] in split_seqs]
    long_costs = [inference_row_cost(r, chunks[r[0]], n) for r in long_rows]
    groups: dict[int, list] = {}
    memo: dict[tuple[int, int, int, int], int] = {}
    for row in rows:
        if row[0] in split_seqs:
            continue
        key = (row[0], row[2], row[3], row[1] if row[3] < 0 else 0)
        cost = memo.get(key)
        if cost is None:
            cost = memo[key] = inference_row_cost(row, chunks[row[0]], n)
        groups.setdefault(cost, []).append(row)
    lists = inference_lists(
        long_rows,
        long_costs,
        [(cost, groups[cost]) for cost in sorted(groups, reverse=True)],
        grid,
    )
    offsets = [0, *itertools.accumulate(len(x) for x in lists)]
    return [r for x in lists for r in x], offsets


# ---- END INFERENCE PLACEMENT ----


def launch_split(rows: Sequence[tuple[int, int, int, int]], arch: str) -> bool:
    """Whether the plan of ``arch`` launches the split build for its ``split_table`` rows: iff some (sequence, head)
    has more than one row (a planned cut, or a V8-BAL row) or a row has span 0 (an empty sequence's copy, a path of
    the split build only).  No Blackwell arch launches the split build for every plan."""

    if arch not in V6_ARCHES:
        raise ValueError(f"the Blackwell prefill plan covers {V6_ARCHES}, got {arch!r}")
    seen = set()
    for seq, head, _cut, span in rows:
        if span <= 0 or (seq, head) in seen:
            return True
        seen.add((seq, head))
    return False


class V6Plan(NamedTuple):
    """One launch of the Blackwell prefill body: the item ``rows`` in table order, the build (``split``: the
    ``split_items`` build, else the block build), the persistent ``grid`` (at most the SM count) and, for the split
    build, the ``grid + 1`` offsets of the CTAs' row lists (CTA ``b`` runs rows ``offsets[b] .. offsets[b + 1] - 1``)."""

    rows: tuple[tuple[int, int, int, int], ...]
    split: bool
    grid: int
    offsets: tuple[int, ...] | None = None


def prefill_v6_record_plan(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> V6Plan:
    """The plan of record of the Blackwell prefill body for the host offsets ``cu_seqlens`` (``N + 1`` integers;
    ``total_tokens`` defaults to the last) on a device of ``sm_count`` SMs (see the rule above): the ``main``
    candidate of the PLAN CHOOSER."""

    if arch not in V6_ARCHES:
        raise ValueError(f"the Blackwell prefill plan covers {V6_ARCHES}, got {arch!r}")
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    H, n, sm_count = int(num_heads), int(num_householder), int(sm_count)
    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")
    table = split_table(raw, n, H, sm_count, total_tokens=total, arch=arch)
    if launch_split(table, arch):
        grid = max(1, min(sm_count, len(table)))
        rows, offsets = inference_csr(table, raw, n, grid, total_tokens=total)
        # every CTA of the launch is resident at once (one CTA per SM): the seed-done wait of the split build needs it
        assert 1 <= grid <= sm_count, (grid, sm_count)
        return V6Plan(rows=tuple(rows), split=True, grid=grid, offsets=tuple(offsets))
    rows = []
    for seq in range(len(raw) - 1):
        start, end = split_sequence_bounds(raw[seq], raw[seq + 1], total)
        if blocks := -(-(end - start) // 64):
            rows.extend((seq, head, 0, blocks) for head in range(H))
    rows.sort(key=lambda row: (-row[3], row[0], row[1]))
    grid = max(1, min(sm_count, len(rows)))
    assert 1 <= grid <= sm_count, (grid, sm_count)
    return V6Plan(rows=tuple(rows), split=False, grid=grid)


# ---- PLAN CHOOSER (sm_100a / sm_103a: the launch plan of one batch) ----
# A call's plan is the least modeled time among candidate plans of the batch, the plan of record unless another is
# modeled V6_PLAN_MARGIN_PERCENT faster.  Every candidate is a launch of a shipped build that keeps its device
# contract (grid at most the SM count, every row of its table once, each CTA's cut-0 rows first):
#   main      the plan of record (``prefill_v6_record_plan``: the split rules of record, ``inference_csr``)
#   f<F>      the split build over the split rules with the least span F (V6_PLAN_FLOORS) in place of the
#             Blackwell floor 48 of a batch of more than 12 heads (``v6_split_count``; V8-BAL as of record), its
#             unsplit rows by least modeled load (calibrated costs)
#   f<F>nb    the same without V8-BAL (F of V6_PLAN_FLOORS_UNIFORM)
#   exact     the split build without a cut
#   block     the block build (no empty sequence), rows longest first, CTA b running rows b, b + G, ...
#   blockbal  the block build with the rows above the least cost on the CTAs of fewest stride positions (LPT)
# Candidates of the same cut decisions are one candidate (the first name).  The model (V6_PLAN_COSTS, per arch,
# hundredths of a microsecond, fitted to CUDA-graph replays of the shipped programs) is a launch's fixed cost plus its
# most loaded CTA: an item of k chunks costs ``k chunk + item`` and a cut one ``warmup`` more, a 1-token row
# ``decode``; a block row of b blocks ``n b block_chunk + block_item`` (``block_decode`` for 1 token).  The plan is
# a pure function of the host offsets, the head count, n, the SM count, the arch and the token count: it reads no
# device memory (a CUDA graph captures the plan of its batch), and it is cached (V6_PLAN_CACHE_SIZE plans, least
# recently used out).  A sequence's cuts may change with its batch: the inference builds agree with the exact
# recurrence within FlashInfer's tolerance for any cut.
# The model above is the time with the FI default gates, whose cuts saturate within a few chunks.  A cut saturates
# only where its gates decay (the device reads them; the host does not), so a candidate is taken only when it is
# robust: for every warm-up need W -- a cut saturates iff it lies at least W chunks into its sequence and W is at
# most the split builds' warm-up cap V6_PLAN_WARMUP_CAP; W runs over the cuts of the candidate and of the plan of
# record up to the cap, and "no cut saturates" -- its modeled time is at most V6_PLAN_ROBUST_PERCENT above the plan
# of record's (``prefill_v6_plan_scenarios``).  Under a need W a saturated cut's row runs ``max(warmup, W chunk)``
# of warm-up and its chunks up to the next saturated cut of its (sequence, head), a cut-0 row runs up to the first
# saturated cut, and an unsaturated cut's row costs ``item`` (its scan); the block build does not depend on W.
# Candidates are tried in order of modeled time (then the first in order) and the first robust one at least
# V6_PLAN_MARGIN_PERCENT faster than the plan of record is the plan.
V6_PLAN_FLOORS = (48, 36, 24, 18, 12)
V6_PLAN_FLOORS_UNIFORM = (48, 24, 12)
V6_PLAN_MARGIN_PERCENT = 3
V6_PLAN_ROBUST_PERCENT = 5
V6_PLAN_WARMUP_CAP = 128  # chunks: the ``split_warmup`` of the Blackwell split builds
V6_PLAN_CACHE_SIZE = 256  # plans (and launches) kept per cache
V6_PLAN_CACHE_ROWS = 1 << 20  # plan rows kept in the ``prefill_v6_plan`` cache
V6_LAUNCH_CACHE_BYTES = (
    64 << 20
)  # item-table bytes kept in the ``prefill_v6_launch`` cache
V6_PLAN_COSTS = {
    "sm_100a": {
        "chunk": 158,
        "warmup": 846,
        "item": 423,
        "decode": 763,
        "fixed": 882,
        "block_chunk": 159,
        "block_item": 305,
        "block_decode": 482,
        "block_fixed": 756,
    },
    "sm_103a": {
        "chunk": 147,
        "warmup": 796,
        "item": 256,
        "decode": 714,
        "fixed": 874,
        "block_chunk": 146,
        "block_item": 458,
        "block_decode": 454,
        "block_fixed": 458,
    },
}
V6_PLAN_SPLIT_SPECS = (
    *((f"f{F}", F, True) for F in V6_PLAN_FLOORS),
    *((f"f{F}nb", F, False) for F in V6_PLAN_FLOORS_UNIFORM),
    ("exact", None, False),
)
V6_PLAN_CANDIDATES = (
    *(name for name, _F, _bal in V6_PLAN_SPLIT_SPECS),
    "main",
    "block",
    "blockbal",
)


@functools.lru_cache(maxsize=65536)
def v6_split_count(
    chunks: int, H: int, n: int, sm_count: int, span_min: int
) -> tuple[int, int]:
    """``split_count`` (auto) with the least span ``span_min`` in place of ``split_span_min(H, arch)``."""
    if chunks <= 0:
        return 1, 0
    if chunks < SPLIT_MIN_CHUNKS:
        return 1, chunks
    best = None
    for p in range(1, SPLIT_MAX + 1):
        span = chunks if p == 1 else split_span(chunks, p, n)
        if p > 1 and span < span_min:
            break
        p_eff = -(-chunks // span)
        waves = -(-(H * p_eff) // sm_count)
        cost = waves * ((span + 1) * SPLIT_COST_CHUNK + SPLIT_COST_ITEM)
        if best is None or cost < best[0]:
            best = (cost, p_eff, span)
    return best[1], best[2]


@functools.lru_cache(maxsize=65536)
def v6_split_bal_slot(chunks: int, H: int, n: int, arch: str, span_min: int) -> int:
    """``split_bal_slot`` (auto) whose uniform-cut comparison takes the least span ``span_min``."""
    if arch not in SPLIT_BAL_SLOTS or chunks < SPLIT_MIN_CHUNKS:
        return 0
    align = split_align(n)
    padded = align * (-(-chunks // align))
    slots = SPLIT_BAL_SLOTS[arch]
    slot = align * (-(-(-(-(H * padded) // slots)) // align))  # noqa: B002
    if not SPLIT_BAL_MIN_SLOT <= slot < padded <= (SPLIT_MAX - 1) * slot:
        return 0
    p, _span = v6_split_count(chunks, H, n, slots, span_min)
    return slot if 2 * slots <= H * p else 0


def _v6_count_scan(chunks, H, n, sm_count, span_min):
    """``(span, p_eff, cost)`` of ``split_count``'s scan over ``p = 1 ..`` while the span is at least
    ``span_min``."""
    out = []
    for p in range(1, SPLIT_MAX + 1):
        span = chunks if p == 1 else split_span(chunks, p, n)
        if p > 1 and span < span_min:
            break
        p_eff = -(-chunks // span)
        out.append(
            (
                span,
                p_eff,
                -(-(H * p_eff) // sm_count)
                * ((span + 1) * SPLIT_COST_CHUNK + SPLIT_COST_ITEM),
            )
        )
    return out


def _v6_scan_best(scan, span_min):
    """``split_count``'s pick from a scan: the least cost over the prefix of spans at least ``span_min`` (the
    first on a tie)."""
    best = None
    for at, (span, p_eff, cost) in enumerate(scan):
        if at and span < span_min:
            break
        if best is None or cost < best[2]:
            best = (span, p_eff, cost)
    return best[1], best[0]


@functools.lru_cache(maxsize=65536)
def _v6_decisions(chunks, H, n, sm_count, arch):
    """The cut of a sequence of ``chunks`` chunks under each split candidate (V6_PLAN_SPLIT_SPECS order): ``(0,
    slot)`` for V8-BAL, else ``(p, span)``.  One scan of ``split_count`` (and one over the V8-BAL slots of the arch)
    serves every floor."""
    if chunks < SPLIT_MIN_CHUNKS:
        return (((1, chunks) if chunks > 0 else (1, 0)),) * len(V6_PLAN_SPLIT_SPECS)
    record = split_span_min(H, arch)
    least = min(
        record,
        *(floor for _name, floor, _bal in V6_PLAN_SPLIT_SPECS if floor is not None),
    )
    scan = _v6_count_scan(chunks, H, n, sm_count, least)
    slot = 0
    if arch in SPLIT_BAL_SLOTS:
        align = split_align(n)
        padded = align * (-(-chunks // align))
        slots = SPLIT_BAL_SLOTS[arch]
        slot = align * (-(-(-(-(H * padded) // slots)) // align))  # noqa: B002
        if not SPLIT_BAL_MIN_SLOT <= slot < padded <= (SPLIT_MAX - 1) * slot:
            slot = 0
    bal_scan = None
    if slot:
        slots = SPLIT_BAL_SLOTS[arch]
        bal_scan = (
            scan if sm_count == slots else _v6_count_scan(chunks, H, n, slots, least)
        )
    out = []
    for _name, floor, bal in V6_PLAN_SPLIT_SPECS:
        if floor is None:
            out.append((1, chunks))
            continue
        span_min = min(floor, record)
        if (
            bal
            and slot
            and 2 * SPLIT_BAL_SLOTS[arch] <= H * _v6_scan_best(bal_scan, span_min)[0]
        ):
            out.append((0, slot))
            continue
        out.append(_v6_scan_best(scan, span_min))
    return tuple(out)


def _v6_split_row_cost(row, chunks: int, tokens: int, n: int, costs) -> int:
    """The modeled cost of a split-build row (V6_PLAN_COSTS ``costs``) of a ``tokens``-token sequence."""
    _seq, head, cut, span = row
    if span == 0:  # an empty sequence's state copy
        return costs["item"]
    if span < 0:
        if cut >= chunks:  # a pad row
            return 0
        align = split_align(n)
        padded = align * (-(-chunks // align))
        at = head * padded + cut
        end = min((at // -span + 1) * -span - head * padded, chunks)
    else:
        end = min(cut + span, chunks)
    if tokens == 1:
        return costs["decode"]
    return (
        max(end - cut, 0) * costs["chunk"]
        + costs["item"]
        + (costs["warmup"] if cut > 0 else 0)
    )


def _v6_block_row_cost(tokens: int, n: int, costs) -> int:
    if tokens == 1:
        return costs["block_decode"]
    return n * (-(-tokens // 64)) * costs["block_chunk"] + costs["block_item"]


@functools.lru_cache(maxsize=4096)
def _v6_cut_shape(chunks, decision, H, n, arch):
    """The rows of a sequence (index 0) cut by ``decision`` in table order, and per row its chunk count and kind
    (0: cut 0, 1: cut > 0, -1: a pad row)."""
    p, span = decision
    if p == 0:  # split_bal_rows of sequence 0, with each piece's chunk count
        slot = span
        align = split_align(n)
        padded = align * (-(-chunks // align))
        top = H * padded
        first, second = [], []
        for k in range(SPLIT_BAL_SLOTS[arch]):
            lo, hi = k * slot, min((k + 1) * slot, top)
            pieces = []
            while lo < hi:
                head, cut = divmod(lo, padded)
                end = min(hi, (head + 1) * padded)
                if cut < chunks:
                    pieces.append(
                        ((0, head, cut, -slot), min(end - head * padded, chunks) - cut)
                    )
                lo = end
            pad = ((0, min(k * slot // padded, H - 1), chunks, -slot), 0)
            if len(pieces) == 2 and pieces[1][0][2] == 0:
                pieces.reverse()
            first.append(pieces[0] if pieces else pad)
            second.append(pieces[1] if len(pieces) > 1 else pad)
        both = first + second
        rows = tuple(row for row, _k in both)
        ks = tuple(k for _row, k in both)
        kinds = tuple(-1 if row[2] >= chunks else 1 if row[2] else 0 for row in rows)
        return rows, ks, kinds
    rows = tuple((0, head, j * span, span) for j in range(p) for head in range(H))
    ks, kinds = [], []
    for j in range(p):
        ks += [max(min(span, chunks - j * span), 0)] * H
        kinds += [1 if j else 0] * H
    return rows, tuple(ks), tuple(kinds)


@functools.lru_cache(maxsize=4096)
def _v6_cut_costs(chunks, decision, H, n, arch, record):
    """The calibrated (or, ``record``, the record) costs of the rows of :func:`_v6_cut_shape`."""
    _rows, ks, kinds = _v6_cut_shape(chunks, decision, H, n, arch)
    if record:
        a, item, warm = 1, INFERENCE_ITEM_COST, INFERENCE_WARMUP_COST
    else:
        c = V6_PLAN_COSTS[arch]
        a, item, warm = c["chunk"], c["item"], c["warmup"]
    return tuple(
        0 if kind < 0 else k * a + item + (warm if kind else 0)
        for k, kind in zip(ks, kinds, strict=False)
    )


@functools.lru_cache(maxsize=4096)
def _v6_cut_stats(chunks, decision, H, n, arch):
    """(rows, total, largest) calibrated cost of the rows of a cut sequence (a uniform cut from its spans alone)."""
    p, span = decision
    if p == 0:
        costs = _v6_cut_costs(chunks, decision, H, n, arch, False)
        return len(costs), sum(costs), max(costs)
    c = V6_PLAN_COSTS[arch]
    each = [
        max(min(span, chunks - j * span), 0) * c["chunk"]
        + c["item"]
        + (c["warmup"] if j else 0)
        for j in range(p)
    ]
    return H * p, H * sum(each), max(each)


def _v6_runs_loads(runs, grid):
    """Per CTA of ``grid``, the cost of rows laid out round-robin from CTA 0 (row ``i`` on CTA ``i % grid``), given
    as ``(rows, cost)`` runs in row order."""
    diff, base, pos = [0] * (grid + 1), 0, 0
    for rows, cost in runs:
        rounds, rest = divmod(rows, grid)
        base += rounds * cost
        first = pos % grid
        if rest:
            last = first + rest
            diff[first] += cost
            if last <= grid:
                diff[last] -= cost
            else:
                diff[grid] -= cost
                diff[0] += cost
                diff[last - grid] -= cost
        pos += rows
    return [base + x for x in itertools.accumulate(diff[:grid])]


@functools.lru_cache(maxsize=4096)
def _v6_cut_loads(chunks, decision, H, n, arch, grid, record):
    """Per CTA of ``grid``, the calibrated (or, ``record``, the record) cost of the cut rows of one sequence placed
    from CTA 0 (row ``i`` on CTA ``i % grid``)."""
    p, span = decision
    if p:  # a uniform cut: H rows of one cost per cut
        if record:
            a, item, warm = 1, INFERENCE_ITEM_COST, INFERENCE_WARMUP_COST
        else:
            c = V6_PLAN_COSTS[arch]
            a, item, warm = c["chunk"], c["item"], c["warmup"]
        runs = [
            (H, max(min(span, chunks - j * span), 0) * a + item + (warm if j else 0))
            for j in range(p)
        ]
        return _v6_runs_loads(runs, grid)
    return _v6_rr_loads(_v6_cut_costs(chunks, decision, H, n, arch, record), grid)


def _v6_rr_loads(costs, grid):
    """Per CTA of ``grid``, the sum of ``costs[i]`` over the rows ``i`` it runs round-robin (``i % grid``)."""
    count = len(costs)
    if count <= grid:
        return list(costs) + [0] * (grid - count)
    full, rest = divmod(count, grid)
    loads = [
        sum(cta)
        for cta in zip(
            *(costs[at : at + grid] for at in range(0, full * grid, grid)), strict=False
        )
    ]
    for b in range(rest):
        loads[b] += costs[full * grid + b]
    return loads


def _rotated(values, shift):
    return values[-shift:] + values[:-shift] if shift else values


class _V6Batch:
    """The host view of one call: lengths, chunks and the sequences of each length (ascending)."""

    __slots__ = (
        "H",
        "n",
        "sm",
        "arch",
        "costs",
        "lens",
        "chunks",
        "by_len",
        "chunk_of",
        "empty",
    )

    def __init__(self, cu, H, n, sm, arch, total):
        self.H, self.n, self.sm, self.arch = H, n, sm, arch
        self.costs = V6_PLAN_COSTS[arch]
        self.lens, self.by_len = [], {}
        start = min(max(int(cu[0]), 0), total)
        for q in range(len(cu) - 1):
            raw = int(cu[q + 1])
            end = min(max(raw, start), total)
            self.lens.append(end - start)
            self.by_len.setdefault(end - start, []).append(q)
            start = min(max(raw, 0), total)
        self.chunk_of = {length: split_chunks(length, n) for length in self.by_len}
        self.chunks = [self.chunk_of[length] for length in self.lens]
        self.empty = 0 in self.by_len


class _V6Split:
    """A split-build candidate: cut decisions per chunk count, rows placed by calibrated or (``record``) record
    costs.  ``bound()`` is a lower bound of the modeled time; ``cost()`` the modeled time; ``plan()`` the launch."""

    def __init__(self, name, bt, decisions, record=False):
        self.name, self.bt, self.dec, self.record = name, bt, decisions, record
        H, costs, n, arch = bt.H, bt.costs, bt.n, bt.arch
        self.long = []  # cut sequences, ascending
        self.short = {}  # placement cost -> [(sequences, calibrated cost)]
        rows = total = top = 0
        for length, seqs in bt.by_len.items():
            chunks = bt.chunk_of[length]
            decision = decisions[chunks]
            if decision[0] != 1:
                self.long.extend(seqs)
                count, sum_, high = _v6_cut_stats(chunks, decision, H, n, arch)
                rows += count * len(seqs)
                total += sum_ * len(seqs)
                top = max(top, high)
                continue
            if length == 0:
                cc = costs["item"]
            elif length == 1:
                cc = costs["decode"]
            else:
                cc = chunks * costs["chunk"] + costs["item"]
            pc = chunks + INFERENCE_ITEM_COST if record else cc
            self.short.setdefault(pc, []).append((seqs, cc))
            rows += H * len(seqs)
            total += H * len(seqs) * cc
            top = max(top, cc)
        if len(self.long) > 1:
            self.long.sort()
        self.items = rows
        self.grid = max(1, min(bt.sm, rows))
        self._bound = costs["fixed"] + max(top, -(-total // self.grid))

    def bound(self):
        return self._bound

    def cost(self):
        bt, G, H = self.bt, self.grid, self.bt.H
        cloads = ploads = None
        at = 0
        for q in self.long:
            chunks = bt.chunks[q]
            decision = self.dec[chunks]
            shift = at % G
            vec = _rotated(
                _v6_cut_loads(chunks, decision, H, bt.n, bt.arch, G, False), shift
            )
            cloads = vec if cloads is None else list(map(operator.add, cloads, vec))
            if self.record:
                vec = _rotated(
                    _v6_cut_loads(chunks, decision, H, bt.n, bt.arch, G, True), shift
                )
                ploads = vec if ploads is None else list(map(operator.add, ploads, vec))
            at += _v6_cut_stats(chunks, decision, H, bt.n, bt.arch)[0]
        if cloads is None:
            cloads = [0] * G
            ploads = [0] * G if self.record else None
        cloads = list(cloads)
        ploads = list(ploads) if self.record else cloads
        for pc in sorted(self.short, reverse=True):
            groups = self.short[pc]
            k = H * sum(len(seqs) for seqs, _cc in groups)
            ccs = {cc for _seqs, cc in groups}
            if len(ccs) == 1:
                cc = ccs.pop()
                counts = lpt_counts(ploads, pc, k)
                if self.record:
                    ploads = [
                        load + x * pc for load, x in zip(ploads, counts, strict=False)
                    ]
                    cloads = [
                        load + x * cc for load, x in zip(cloads, counts, strict=False)
                    ]
                else:
                    cloads = ploads = [
                        load + x * cc for load, x in zip(cloads, counts, strict=False)
                    ]
                continue
            # one placement cost, several calibrated costs: follow the rows
            per_seq = sorted((q, cc) for seqs, cc in groups for q in seqs)
            head, phase, full, tail = lpt_rounds(ploads, pc, k)
            slots = head + phase * full + tail
            costs = [cc for _q, cc in per_seq for _head in range(H)]
            for b, cc in zip(slots, costs, strict=False):
                cloads[b] += cc
                ploads[b] += pc
        self.loads = cloads
        return bt.costs["fixed"] + max(cloads)

    def plan(self):
        if getattr(self, "_plan", None) is None:
            self._plan = self._build()
        return self._plan

    def _build(self):
        bt, G, H, n, arch = self.bt, self.grid, self.bt.H, self.bt.n, self.bt.arch
        long_rows, long_costs, flags = [], [], []
        for q in self.long:
            chunks = bt.chunks[q]
            decision = self.dec[chunks]
            long_rows.extend(_v6_cut_rows(q, chunks, decision, H, n, arch))
            long_costs.extend(_v6_cut_costs(chunks, decision, H, n, arch, self.record))
            flags.extend(_v6_cut_flags(chunks, decision, H, n, arch))
        short = []
        for pc in sorted(self.short, reverse=True):
            seqs = [q for group, _cc in self.short[pc] for q in group]
            if len(self.short[pc]) > 1:
                seqs.sort()
            short.append(
                (
                    pc,
                    list(
                        itertools.chain.from_iterable(
                            _v6_head_rows(q, bt.chunks[q], H) for q in seqs
                        )
                    ),
                )
            )
        lists = inference_lists(long_rows, long_costs, short, G, flags)
        offsets = (0, *itertools.accumulate(map(len, lists)))
        return V6Plan(
            rows=tuple(itertools.chain.from_iterable(lists)),
            split=True,
            grid=G,
            offsets=offsets,
        )


@functools.lru_cache(maxsize=4096)
def _v6_cut_rows(seq, chunks, decision, H, n, arch):
    """The rows of sequence ``seq`` cut by ``decision``, in table order."""
    rows = _v6_cut_shape(chunks, decision, H, n, arch)[0]
    return rows if seq == 0 else tuple((seq, h, c, sp) for _s, h, c, sp in rows)


@functools.lru_cache(maxsize=4096)
def _v6_cut_flags(chunks, decision, H, n, arch):
    """Per row of a cut sequence: 1 for a row of cut > 0 (a pad row included), else 0."""
    return tuple(
        1 if kind else 0 for kind in _v6_cut_shape(chunks, decision, H, n, arch)[2]
    )


@functools.lru_cache(maxsize=4096)
def _v6_head_rows(seq, span, H):
    """The rows ``(seq, head, 0, span)`` of an unsplit sequence, heads in order."""
    return tuple((seq, h, 0, span) for h in range(H))


class _V6Block:
    """A block-build candidate: CTA b runs table positions b, b + G, ...; ``balanced`` moves the rows above the least
    cost to the CTAs of the fewest positions (LPT over the stride layout's per-CTA capacities)."""

    def __init__(self, name, bt, balanced, layout=None):
        self.name, self.bt, self.balanced = name, bt, balanced
        H, n, costs = bt.H, bt.n, bt.costs
        if layout is None:
            order = []  # sequences in row order: blocks descending, then index
            runs = []  # [rows, cost] of equal cost, in row order
            by_blocks: dict[int, list] = {}
            for length, seqs in bt.by_len.items():
                if length > 0:
                    by_blocks.setdefault(-(-length // 64), []).extend(
                        (q, length) for q in seqs
                    )
            total = top = 0
            for blocks in sorted(by_blocks, reverse=True):
                group = by_blocks[blocks]
                if len(group) > 1:
                    group.sort()
                for q, length in group:
                    order.append(q)
                    cost = _v6_block_row_cost(length, n, costs)
                    if runs and runs[-1][1] == cost:
                        runs[-1][0] += H
                    else:
                        runs.append([H, cost])
                    total += H * cost
                    top = max(top, cost)
            layout = (order, runs, total, top)
        self.layout = layout
        self.order, self.runs, total, top = layout
        self.items = len(self.order) * H
        self.grid = max(1, min(bt.sm, self.items))
        self._bound = costs["block_fixed"] + max(top, -(-total // self.grid))
        self.ok = True
        if balanced:
            least = min((cost for _rows, cost in self.runs), default=0)
            big = sum(rows for rows, cost in self.runs if cost > least)
            # nothing to balance, or many long rows (the stride order is as good)
            self.ok = 0 < big <= 4 * self.grid

    def bound(self):
        return self._bound

    def cost(self):
        G, runs = self.grid, self.runs
        if not self.balanced:
            self.loads = _v6_runs_loads(runs, G)
            return self.bt.costs["block_fixed"] + max(self.loads)
        R = self.items
        least = min(cost for _rows, cost in runs)
        self.cap = [R // G + (1 if b < R % G else 0) for b in range(G)]
        used, loads = [0] * G, [0] * G
        heap = [(self.cap[b] * least, b) for b in range(G)]
        heapq.heapify(heap)
        self.assign = []  # the CTA of each row above the least cost, rows by cost descending (stable)
        for rows, cost in sorted(
            ((r, c) for r, c in runs if c > least), key=lambda rc: -rc[1]
        ):
            for _ in range(rows):
                _key, b = heapq.heappop(heap)
                used[b] += 1
                loads[b] += cost
                self.assign.append(b)
                if used[b] < self.cap[b]:
                    heapq.heappush(
                        heap, (loads[b] + (self.cap[b] - used[b]) * least, b)
                    )
        self.loads = [
            load + (self.cap[b] - used[b]) * least for b, load in enumerate(loads)
        ]
        self.least = least
        return self.bt.costs["block_fixed"] + max(self.loads)

    def plan(self):
        if getattr(self, "_plan", None) is None:
            self._plan = self._build()
        return self._plan

    def _build(self):
        bt, G, H = self.bt, self.grid, self.bt.H
        rows = [(q, h, 0, -(-bt.lens[q] // 64)) for q in self.order for h in range(H)]
        if not self.balanced:
            return V6Plan(rows=tuple(rows), split=False, grid=G)
        if not hasattr(self, "assign"):
            self.cost()
        n, costs = bt.n, bt.costs
        cost_of = [_v6_block_row_cost(bt.lens[r[0]], n, costs) for r in rows]
        big = sorted(
            (i for i in range(len(rows)) if cost_of[i] > self.least),
            key=lambda i: -cost_of[i],
        )
        lists = [[] for _ in range(G)]
        for i, b in zip(big, self.assign, strict=False):
            lists[b].append(rows[i])
        rest = iter(rows[i] for i in range(len(rows)) if cost_of[i] <= self.least)
        for b in range(G):
            while len(lists[b]) < self.cap[b]:
                lists[b].append(next(rest))
        rounds = max(len(x) for x in lists)
        placed = [
            lists[b][r] for r in range(rounds) for b in range(G) if r < len(lists[b])
        ]
        return V6Plan(rows=tuple(placed), split=False, grid=G)


def _v6_candidates(bt):
    """The distinct candidates of a batch in V6_PLAN_CANDIDATES order and ``alias`` (every applicable name -> the
    name of the distinct candidate of its plan)."""
    distinct = sorted(set(bt.chunk_of.values()))
    per = [_v6_decisions(c, bt.H, bt.n, bt.sm, bt.arch) for c in distinct]
    out, seen, alias = [], {}, {}
    for at, (name, _floor, _bal) in enumerate(V6_PLAN_SPLIT_SPECS):
        key = tuple(d[at] for d in per)
        if key in seen:
            alias[name] = seen[key]
            continue
        seen[key] = alias[name] = name
        out.append(_V6Split(name, bt, dict(zip(distinct, key, strict=False))))
    record = next(c for c in out if c.name == alias["f48"]).dec
    if bt.empty or any(
        record[c][0] != 1 for c in distinct
    ):  # launch_split of the table of record
        out.append(_V6Split("main", bt, record, record=True))
        alias["main"] = "main"
    else:
        alias["main"] = "block"
    if not bt.empty:  # the block build runs no empty sequence
        block = _V6Block("block", bt, False)
        out.append(block)
        alias["block"] = "block"
        balanced = _V6Block("blockbal", bt, True, block.layout)
        if balanced.ok:
            out.append(balanced)
        alias["blockbal"] = "blockbal" if balanced.ok else "block"
    return out, alias


def _v6_plan_ctas(plan):
    """Per CTA of ``plan``, the rows it runs: the split build's row lists, the block build's stride positions."""
    rows, G = plan.rows, plan.grid
    if plan.split:
        o = plan.offsets
        return [rows[o[b] : o[b + 1]] for b in range(G)]
    return [rows[b::G] for b in range(G)]


def _v6_cut_map(plan, bt):
    """``(seq, head) -> ascending cuts`` of every (sequence, head) of more than one row of a split plan (pad rows
    and state copies excluded): the cut sequences."""
    cuts: dict[tuple[int, int], list[int]] = {}
    if plan.split:
        for seq, head, cut, span in plan.rows:
            if span != 0 and cut < bt.chunks[seq]:
                cuts.setdefault((seq, head), []).append(cut)
    return {key: sorted(v) for key, v in cuts.items() if len(v) > 1}


def _v6_scenario_time(plan, bt, cut_map, W):
    """The modeled time of ``plan`` when a cut saturates iff it lies at least ``W`` chunks into its sequence and
    ``W`` is at most V6_PLAN_WARMUP_CAP (``W`` None: no cut saturates); see the PLAN CHOOSER."""
    c = bt.costs
    if not plan.split:
        loads = [
            sum(_v6_block_row_cost(bt.lens[r[0]], bt.n, c) for r in cta)
            for cta in _v6_plan_ctas(plan)
        ]
        return c["block_fixed"] + max(loads, default=0)
    live = W is not None and W <= V6_PLAN_WARMUP_CAP
    warm = max(c["warmup"], W * c["chunk"]) if live else 0
    top = 0
    for cta in _v6_plan_ctas(plan):
        load = 0
        for row in cta:
            seq, head, cut, span = row
            cuts = cut_map.get((seq, head))
            chunks = bt.chunks[seq]
            if cuts is None or span == 0 or cut >= chunks:
                load += _v6_split_row_cost(row, chunks, bt.lens[seq], bt.n, c)
                continue
            if cut and not (live and cut >= W):
                load += c["item"]  # an unsaturated cut: its scan
                continue
            end = chunks
            if live:
                at = bisect.bisect_right(cuts, max(cut, W - 1))
                if at < len(cuts):
                    end = cuts[at]
            load += (end - cut) * c["chunk"] + c["item"] + (warm if cut else 0)
        top = max(top, load)
    return c["fixed"] + top


def _v6_scenarios(plan: V6Plan, ref: V6Plan, bt) -> list[tuple[int | None, int, int]]:
    """``(W, time of plan, time of ref)`` per warm-up need of the robustness rule (PLAN CHOOSER): every cut of
    either plan up to V6_PLAN_WARMUP_CAP, then None (no cut saturates); modeled hundredths of a microsecond."""
    maps = (_v6_cut_map(plan, bt), _v6_cut_map(ref, bt))
    needs = sorted(
        {
            cut
            for m in maps
            for cuts in m.values()
            for cut in cuts
            if 0 < cut <= V6_PLAN_WARMUP_CAP
        }
    )
    return [
        (
            W,
            _v6_scenario_time(plan, bt, maps[0], W),
            _v6_scenario_time(ref, bt, maps[1], W),
        )
        for W in (*needs, None)
    ]


def _v6_robust(cand, ref, bt) -> bool:
    return all(
        100 * mine <= (100 + V6_PLAN_ROBUST_PERCENT) * theirs
        for _W, mine, theirs in _v6_scenarios(cand.plan(), ref.plan(), bt)
    )


def _v6_choose(bt):
    """The robust candidate of least modeled time (then the first in order) among those V6_PLAN_MARGIN_PERCENT
    below the plan of record's time, else the plan of record.  Candidates whose lower bound cannot come below that
    limit are not modeled, which never changes the pick."""
    cands, alias = _v6_candidates(bt)
    ref = next(c for c in cands if c.name == alias["main"])
    limit = (100 - V6_PLAN_MARGIN_PERCENT) * ref.cost()
    fast = []
    for at, cand in enumerate(cands):
        if cand is ref or 100 * cand.bound() >= limit:
            continue
        cost = cand.cost()
        if 100 * cost < limit:
            fast.append((cost, at))
    for _cost, at in sorted(fast):
        if _v6_robust(cands[at], ref, bt):
            return cands[at]
    return ref


def _v6_batch(cu_seqlens, num_heads, num_householder, sm_count, arch, total_tokens):
    if arch not in V6_ARCHES:
        raise ValueError(f"the Blackwell prefill plan covers {V6_ARCHES}, got {arch!r}")
    sm_count = int(sm_count)
    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")
    if type(cu_seqlens) is tuple:
        cu = cu_seqlens
    elif isinstance(cu_seqlens, torch.Tensor):
        cu = tuple(cu_seqlens.tolist())
    else:
        cu = tuple(cu_seqlens)
    total = int(cu[-1]) if total_tokens is None else int(total_tokens)
    return cu, int(num_heads), int(num_householder), sm_count, arch, total


def _v6_check_grid(grid: int, sm_count: int) -> None:
    # every CTA of the launch is resident at once (one CTA per SM): the seed-done wait of the split build needs it
    if not 1 <= grid <= sm_count:
        raise RuntimeError(
            f"Blackwell prefill plan grid {grid} outside 1..{sm_count} (the SM count)"
        )


def _v6_reference(cu, H, n, sm_count, arch, total):
    """The PLAN CHOOSER's pick and its plan, in Python."""
    cand = _v6_choose(_V6Batch(cu, H, n, sm_count, arch, total))
    plan = cand.plan()
    _v6_check_grid(plan.grid, sm_count)
    return cand.name, plan


def prefill_v6_plan_reference(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> V6Plan:
    """The launch plan of the PLAN CHOOSER computed in Python (uncached): the reference of the native chooser."""

    return _v6_reference(
        *_v6_batch(cu_seqlens, num_heads, num_householder, sm_count, arch, total_tokens)
    )[1]


def prefill_v6_plan(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> V6Plan:
    """The launch plan of the Blackwell prefill body for the host offsets ``cu_seqlens`` (``N + 1`` integers;
    ``total_tokens`` defaults to the last) on a device of ``sm_count`` SMs: the candidate the PLAN CHOOSER picks
    (the native chooser when it builds, else the Python reference; cached, see NATIVE PLAN CHOOSER)."""

    # the key: a tensor's offsets as bytes, a sequence's as a tuple (width 0)
    data: bytes | tuple[int, ...]
    if isinstance(cu_seqlens, torch.Tensor):
        data, width = _v6_offset_bytes(cu_seqlens)
        if total_tokens is None:
            total_tokens = int.from_bytes(data[-width:], "little", signed=True)
    else:
        data, width = (
            (cu_seqlens if type(cu_seqlens) is tuple else tuple(cu_seqlens)),
            0,
        )
        if total_tokens is None:
            total_tokens = data[-1]
    key = (
        data,
        width,
        int(num_heads),
        int(num_householder),
        int(sm_count),
        arch,
        int(total_tokens),
    )
    plan = _V6_PLANS.lookup(key)
    if plan is not None:
        return plan
    _data, _width, H, n, sm, _arch, total = key
    if width == 0:
        data, width = array.array("q", data).tobytes(), 8
    if arch not in V6_ARCHES:
        raise ValueError(f"the Blackwell prefill plan covers {V6_ARCHES}, got {arch!r}")
    if sm <= 0:
        raise ValueError(f"sm_count must be positive, got {sm}")
    native = _v6_native_module()
    if native is None:
        cu = tuple(array.array("q" if width == 8 else "i", data))
        plan = _v6_reference(cu, H, n, sm, arch, total)[1]
    else:
        out = native.v6_launch(
            data, width, H, n, sm, total, _v6_native_params(arch, H), 1
        )
        _cand, split, grid, count, _words = _V6_NATIVE_HEADER.unpack_from(out)
        words = array.array("i", out[4 * V6_NATIVE_HEAD :])
        it = iter(words[: 4 * count])
        rows = tuple(zip(it, it, it, it, strict=False))
        offsets = (
            tuple(words[4 * max(count, 1) : 4 * max(count, 1) + grid + 1])
            if split
            else None
        )
        plan = V6Plan(rows=rows, split=bool(split), grid=grid, offsets=offsets)
    _V6_PLANS.put(key, plan, max(len(plan.rows), 1))
    return plan


def prefill_v6_plan_choice(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> str:
    """The name of the candidate :func:`prefill_v6_plan` launches for the batch (computed afresh)."""

    bt = _V6Batch(
        *_v6_batch(cu_seqlens, num_heads, num_householder, sm_count, arch, total_tokens)
    )
    return _v6_choose(bt).name


def prefill_v6_plan_candidates(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> dict[str, tuple[str, int]]:
    """Every applicable candidate name of the batch -> (the distinct candidate of its plan, its modeled time in
    hundredths of a microsecond), every candidate modeled."""

    bt = _V6Batch(
        *_v6_batch(cu_seqlens, num_heads, num_householder, sm_count, arch, total_tokens)
    )
    cands, alias = _v6_candidates(bt)
    cost = {c.name: c.cost() for c in cands}
    return {
        name: (alias[name], cost[alias[name]])
        for name in V6_PLAN_CANDIDATES
        if name in alias
    }


def prefill_v6_plan_scenarios(
    name: str,
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> list[tuple[int | None, int, int]]:
    """The robustness scenarios of candidate ``name`` against the plan of record (see the PLAN CHOOSER): ``(W,
    modeled time of the candidate, modeled time of the plan of record)`` per warm-up need ``W`` (None: no cut
    saturates), hundredths of a microsecond."""

    bt = _V6Batch(
        *_v6_batch(cu_seqlens, num_heads, num_householder, sm_count, arch, total_tokens)
    )
    cands, alias = _v6_candidates(bt)
    if name not in alias:
        raise ValueError(
            f"candidate {name!r} does not apply to the batch; it takes {sorted(alias)}"
        )
    cand = next(c for c in cands if c.name == alias[name])
    ref = next(c for c in cands if c.name == alias["main"])
    return _v6_scenarios(cand.plan(), ref.plan(), bt)


def prefill_v6_candidate_plan(
    name: str,
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
) -> V6Plan:
    """The launch plan of candidate ``name`` (V6_PLAN_CANDIDATES) for the batch; ValueError when it does not apply
    (a block candidate with an empty sequence)."""

    bt = _V6Batch(
        *_v6_batch(cu_seqlens, num_heads, num_householder, sm_count, arch, total_tokens)
    )
    cands, alias = _v6_candidates(bt)
    if name not in alias:
        raise ValueError(
            f"candidate {name!r} does not apply to the batch; it takes {sorted(alias)}"
        )
    return next(c for c in cands if c.name == alias[name]).plan()


# ---- END PLAN CHOOSER ----


# ---- NATIVE PLAN CHOOSER (the host time of an eager call) ----
# ``csrc/cake_gdp_v6_plan.cpp`` is the PLAN CHOOSER in native code: a line-by-line port of the Python chooser above
# (the reference, ``prefill_v6_plan_reference``) that takes every constant from this file (``_v6_native_params``)
# and returns the launch and the int32 words of its item table at once.  FlashInfer's JIT builds it on first use
# (a host library, no device code); when it cannot be built the Python chooser plans every call (one warning).
# ``prefill_v6_launch`` is what a call launches: the program of the call's state form, the build, the grid, the row
# count and the item words, cached by the call's host offsets (their bytes), H, n, the SM count, the arch, the token
# count and the state form; the cache keeps V6_PLAN_CACHE_SIZE launches and at most V6_LAUNCH_CACHE_BYTES of words
# (least recently used out), the ``prefill_v6_plan`` cache at most V6_PLAN_CACHE_ROWS rows.  ``clear_plan_caches``
# empties every planner cache (a batch the process has not planned before).
V6_NATIVE_SOURCE = "cake_gdp_v6_plan.cpp"
V6_NATIVE_HEAD = (
    5  # header words of the native result: candidate, split, grid, rows, item words
)
_V6_NATIVE_HEADER = struct.Struct(f"<{V6_NATIVE_HEAD}i")
_V6_NATIVE: list = []  # [module or None] once loaded


class _V6Cache(collections.OrderedDict):
    """A least-recently-used cache of at most V6_PLAN_CACHE_SIZE entries whose summed weight stays at most
    ``budget`` (one entry always stays)."""

    def __init__(self, budget: int):
        super().__init__()
        self.budget, self.total = budget, 0
        self.weights: dict[object, int] = {}

    def lookup(self, key):
        value = self.get(key)
        if value is not None:
            self.move_to_end(key)
        return value

    def put(self, key, value, weight: int) -> None:
        self[key] = value
        self.weights[key] = weight
        self.total += weight
        while len(self) > 1 and (
            len(self) > V6_PLAN_CACHE_SIZE or self.total > self.budget
        ):
            old, _value = self.popitem(last=False)
            self.total -= self.weights.pop(old)

    def clear(self) -> None:
        super().clear()
        self.weights.clear()
        self.total = 0


_V6_PLANS = _V6Cache(V6_PLAN_CACHE_ROWS)
_V6_LAUNCHES = _V6Cache(V6_LAUNCH_CACHE_BYTES)


class V6Launch(NamedTuple):
    """One launch of the Blackwell prefill body: the program ``name``, the chooser's candidate, the build (``split``),
    the grid, the item-table row count and the int32 ``words`` of the ``items`` formal (``v6_item_words``)."""

    name: str
    candidate: str
    split: bool
    grid: int
    num_items: int
    words: bytes


def clear_plan_caches() -> None:
    """Empty every cache of the Blackwell launch planner: the launches, the plans and the chooser's memo tables (the
    native library and the programs of each state form, ``_v6_launch_builds``, stay)."""
    _V6_PLANS.clear()
    _V6_LAUNCHES.clear()
    for fn in (
        v6_split_count,
        v6_split_bal_slot,
        _v6_decisions,
        _v6_cut_shape,
        _v6_cut_costs,
        _v6_cut_stats,
        _v6_cut_loads,
        _v6_cut_rows,
        _v6_cut_flags,
        _v6_head_rows,
    ):
        fn.cache_clear()


def _v6_native_path() -> Path | None:
    """The native chooser's source: the ``csrc`` of the tree this package runs from, else of the installed
    FlashInfer."""
    here = Path(__file__).resolve().parents[2] / "csrc" / V6_NATIVE_SOURCE
    if here.is_file():
        return here
    try:
        from flashinfer.jit import env as jit_env
    except ImportError:
        return None
    path = Path(jit_env.FLASHINFER_CSRC_DIR) / V6_NATIVE_SOURCE
    return path if path.is_file() else None


def _v6_native_module():
    """The loaded native chooser, or None when it cannot be built (then the Python chooser plans every call)."""
    if _V6_NATIVE:
        return _V6_NATIVE[0]
    module = None
    try:
        path = _v6_native_path()
        if path is None:
            raise RuntimeError(f"{V6_NATIVE_SOURCE} not found")
        from flashinfer.jit.core import gen_jit_spec

        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        module = gen_jit_spec(f"cake_gdp_v6_plan_{digest}", [path]).build_and_load()
    except Exception as exc:  # noqa: BLE001 -- any build failure keeps the (slower, equal) Python chooser
        warnings.warn(
            f"the native Cake GDP prefill planner is not available ({type(exc).__name__}: {exc}); "
            "the Python chooser plans every call",
            RuntimeWarning,
            stacklevel=2,
        )
    _V6_NATIVE.append(module)
    return module


def _v6_native_param_parts(arch: str) -> tuple[bytes, bytes]:
    """The int64 parameter words of the native chooser around ``split_span_min`` (the layout of ``Param`` in its
    source), from this file's constants."""
    c = V6_PLAN_COSTS[arch]
    head = [
        *(c[k] for k in ("chunk", "warmup", "item", "decode", "fixed")),
        *(c[k] for k in ("block_chunk", "block_item", "block_decode", "block_fixed")),
        V6_PLAN_MARGIN_PERCENT,
        V6_PLAN_ROBUST_PERCENT,
        V6_PLAN_WARMUP_CAP,
        SPLIT_MIN_CHUNKS,
        SPLIT_MAX,
        SPLIT_COST_CHUNK,
        SPLIT_COST_ITEM,
        SPLIT_BAL_MIN_SLOT,
        SPLIT_BAL_SLOTS.get(arch, 0),
        INFERENCE_ITEM_COST,
        INFERENCE_WARMUP_COST,
    ]
    tail = [len(V6_PLAN_SPLIT_SPECS)]
    for _name, floor, bal in V6_PLAN_SPLIT_SPECS:
        tail += [-1 if floor is None else floor, int(bal)]
    return array.array("q", head).tobytes(), array.array("q", tail).tobytes()


_V6_NATIVE_PARAM_PARTS = {arch: _v6_native_param_parts(arch) for arch in V6_PLAN_COSTS}
_V6_INT64 = struct.Struct("<q")


def _v6_native_params(arch: str, H: int) -> bytes:
    """The int64 parameter words of the native chooser for ``H`` heads on ``arch`` (the parts are computed once,
    at import, from this file's constants)."""
    head, tail = _V6_NATIVE_PARAM_PARTS[arch]
    return head + _V6_INT64.pack(split_span_min(H, arch)) + tail


@functools.lru_cache(maxsize=64)
def _v6_launch_builds(arch: str, state_dtype: str, pool: bool):
    """``(split -> program, flags)`` of a state form: flags bit 0 / 1 = the split build reads CTA row offsets /
    the planned offsets (``csr`` / ``plan_guard``), bits 2 / 3 the same for the block build."""
    builds = v6_route_builds(arch, state_dtype, pool)
    flags = 0
    for split, shift in ((True, 0), (False, 2)):
        if split in builds:
            build = MODULES[builds[split]]["build"]
            flags |= (
                int(bool(build.get("csr"))) | 2 * int(bool(build.get("plan_guard")))
            ) << shift
    return builds, flags


def _v6_words(flags: int, plan: V6Plan, cu: Sequence[int]) -> list[int]:
    """``v6_item_words`` for the build flags of ``_v6_launch_builds``."""
    shift = 0 if plan.split else 2
    words = [int(w) for row in (plan.rows or ((0, 0, 0, 0),)) for w in row]
    if flags >> shift & 1:
        if plan.offsets is None:
            raise ValueError("the csr build runs a plan with row offsets")
        words.extend(int(o) for o in plan.offsets)
    if flags >> shift & 2:
        words.extend((len(cu) - 1, *(int(c) for c in cu)))
    return words


def v6_launch_of_plan(
    arch: str,
    plan: V6Plan,
    cu_seqlens: Sequence[int],
    *,
    state_dtype: str = "f32",
    pool: bool = False,
) -> V6Launch:
    """The launch of a given ``plan`` (a candidate's, the plan of record) for the host offsets ``cu_seqlens``, on the
    program of a state form: what ``prefill_v6_launch`` returns for the chooser's plan."""

    builds, flags = _v6_launch_builds(arch, state_dtype, bool(pool))
    if plan.split not in builds:
        raise ValueError(
            f"no Blackwell prefill program on {arch} with split_items={plan.split} for that form"
        )
    words = array.array(
        "i", _v6_words(flags, plan, [int(c) for c in cu_seqlens])
    ).tobytes()
    return V6Launch(
        builds[plan.split], "given", plan.split, plan.grid, len(plan.rows), words
    )


def _v6_offset_bytes(cu_seqlens) -> tuple[bytes, int]:
    """The host offsets as (bytes, element width): a CPU int32 / int64 tensor as is, else int64."""
    if isinstance(cu_seqlens, torch.Tensor):
        width = cu_seqlens.element_size()
        if cu_seqlens.is_cpu and cu_seqlens.is_contiguous():
            return ctypes.string_at(
                cu_seqlens.data_ptr(), cu_seqlens.numel() * width
            ), width
        return cu_seqlens.numpy().tobytes(), width
    return array.array("q", cu_seqlens).tobytes(), 8


def prefill_v6_launch(
    cu_seqlens,
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    arch: str,
    total_tokens: int | None = None,
    state_dtype: str = "f32",
    pool: bool = False,
) -> V6Launch:
    """The launch of a call (see NATIVE PLAN CHOOSER): ``cu_seqlens`` the host offsets as a CPU int32 / int64 tensor
    or a sequence of integers, the state form as in ``selected_v6_module``."""

    data, width = _v6_offset_bytes(cu_seqlens)
    if total_tokens is None:
        total_tokens = int.from_bytes(data[-width:], "little", signed=True)
    H, n, sm, total = (
        int(num_heads),
        int(num_householder),
        int(sm_count),
        int(total_tokens),
    )
    key = (data, width, H, n, sm, arch, total, state_dtype, bool(pool))
    launch = _V6_LAUNCHES.lookup(key)
    if launch is not None:
        return launch
    if arch not in V6_ARCHES:
        raise ValueError(f"the Blackwell prefill plan covers {V6_ARCHES}, got {arch!r}")
    if sm <= 0:
        raise ValueError(f"sm_count must be positive, got {sm}")
    builds, flags = _v6_launch_builds(arch, state_dtype, bool(pool))
    native = _v6_native_module()
    if native is not None:
        out = native.v6_launch(
            data, width, H, n, sm, total, _v6_native_params(arch, H), flags
        )
        cand, split, grid, count, _words = _V6_NATIVE_HEADER.unpack_from(out)
        candidate, split, words = (
            V6_PLAN_CANDIDATES[cand],
            bool(split),
            out[4 * V6_NATIVE_HEAD :],
        )
    else:
        cu = array.array("q" if width == 8 else "i", data).tolist()
        candidate, plan = _v6_reference(tuple(cu), H, n, sm, arch, total)
        split, grid, count = plan.split, plan.grid, len(plan.rows)
        words = array.array("i", _v6_words(flags, plan, cu)).tobytes()
    _v6_check_grid(grid, sm)
    if split not in builds:
        raise ValueError(
            f"no Blackwell prefill program on {arch} with split_items={split}, state_dtype={state_dtype!r} and "
            f"pool={bool(pool)}; the route ships the forms {sorted(v6_state_forms(arch))}"
        )
    launch = V6Launch(builds[split], candidate, split, grid, count, words)
    _V6_LAUNCHES.put(key, launch, len(words))
    return launch


# ---- END NATIVE PLAN CHOOSER ----


def v6_item_words(
    arch: str, plan: V6Plan, cu_seqlens: Sequence[int]
) -> tuple[int, ...]:
    """The int32 words of the ``items`` formal of the program ``plan`` selects on ``arch``: the rows (4 words each;
    one zero row when there is none), then the CTAs' row offsets (a ``csr`` build), then the planned sequence count
    ``N`` and the ``N + 1`` host offsets ``cu_seqlens`` the plan was made from (a ``plan_guard`` build).  The guarded
    program compares those with the device offsets and, when they differ, runs no item and writes NaN to
    ``o[0, 0, :2]`` (one 32-bit word) and, when it stores one, to final state ``[0, 0, 0, 0]``: a batch changed
    after the plan fails loudly."""

    build = MODULES[selected_v6_module(arch, plan)]["build"]
    words = [int(w) for row in (plan.rows or ((0, 0, 0, 0),)) for w in row]
    if build.get("csr"):
        if plan.offsets is None:
            raise ValueError("the csr build runs a plan with row offsets")
        words.extend(int(o) for o in plan.offsets)
    if build.get("plan_guard"):
        cu = [int(c) for c in cu_seqlens]
        words.extend((len(cu) - 1, *cu))
    return tuple(words)


def v6_state_form(name: str) -> tuple[str, bool]:
    """The state form of the Blackwell prefill program ``name``: its state dtype (``"f32"``, ``"bf16"`` or
    ``"f16"``) and whether it addresses slot pools."""

    build = MODULES[name]["build"]
    return str(build.get("state_dtype", "f32")), bool(
        build.get("use_state_indices", False)
    )


def _v6_route_names(arch: str) -> tuple[str, ...]:
    names = []
    for name in ROUTES.get(arch, {}).get(PREFILL_ROUTE, ()):
        record = MODULES[name]
        if record["body"] != V6_BODY:
            continue
        rule = record["select_rule"]
        if not isinstance(rule, dict) or rule.get("kind") != V6_SELECT_KIND:
            raise ValueError(
                f"not a Cake GDP Blackwell prefill selection rule: {rule!r}"
            )
        names.append(name)
    return tuple(names)


def v6_route_builds(
    arch: str, state_dtype: str = "f32", pool: bool = False
) -> dict[bool, str]:
    """``split_items -> program`` of the Blackwell prefill body on ``arch`` for one state form: the programs whose
    build stores ``state_dtype`` (``"f32"``, ``"bf16"`` or ``"f16"``) and addresses slot pools when ``pool``."""

    builds: dict[bool, str] = {}
    for name in _v6_route_names(arch):
        if v6_state_form(name) != (state_dtype, bool(pool)):
            continue
        split = bool(MODULES[name]["select_rule"]["split_items"])
        if split in builds:
            raise ValueError(
                f"two Blackwell prefill programs on {arch} with split_items={split}, "
                f"state_dtype={state_dtype!r} and pool={bool(pool)}"
            )
        builds[split] = name
    return builds


def v6_state_forms(arch: str) -> set[tuple[str, bool]]:
    """The state forms ``(state_dtype, pool)`` the Blackwell prefill programs of ``arch`` ship."""

    return {v6_state_form(name) for name in _v6_route_names(arch)}


def v6_householders(arch: str) -> set[int]:
    """The ``num_householder`` values the Blackwell prefill programs of ``arch`` build for (the body reads ``n`` as
    a build constant)."""

    return {
        int(MODULES[name]["build"]["householders"]) for name in _v6_route_names(arch)
    }


def selected_v6_module(
    arch: str, plan: V6Plan, *, state_dtype: str = "f32", pool: bool = False
) -> str:
    """The program of the Blackwell prefill body on ``arch`` that ``plan`` launches for one state form: the dense
    FP32 programs by default, the slot-pool programs of ``state_dtype`` with ``pool`` (a 16-bit state is a pool
    program only)."""

    builds = v6_route_builds(arch, state_dtype, pool)
    if plan.split not in builds:
        raise ValueError(
            f"no Blackwell prefill program on {arch} with split_items={plan.split}, "
            f"state_dtype={state_dtype!r} and pool={bool(pool)}; the route ships the forms "
            f"{sorted(v6_state_forms(arch))}"
        )
    return builds[plan.split]


# ---- END BLACKWELL V6 LAUNCH PLAN ----


def prefill_split_table(
    cu_seqlens: Sequence[int],
    *,
    num_heads: int,
    num_householder: int,
    sm_count: int,
    total_tokens: int | None = None,
) -> list[tuple[int, int, int, int]] | None:
    """The item table of the ``split_items`` programs for a call whose host knows
    its sequence lengths (``cu_seqlens``: ``N + 1`` host integers), or ``None``
    when every sequence of the call is one item (``p = 1``): the exact programs
    run then, a one-item sequence under a split program being bitwise the single
    pass.  The rule reads each sequence alone, so a sequence
    splits the same in any batch."""

    table = split_table(
        cu_seqlens,
        int(num_householder),
        int(num_heads),
        int(sm_count),
        total_tokens=total_tokens,
    )
    if len(table) == (len(cu_seqlens) - 1) * int(num_heads):
        return None
    return table


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any],
    batch: int,
    heads: int,
    *,
    tiles: int | None = None,
    sm_count: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one module under its recorded grid rule.  ``state_heads`` (decode): ``ceil(B x HV /
    heads_per_cta) x ctas_per_state`` CTAs (``heads_per_cta`` defaults to one: one CTA per batch row and head for the
    lane-owns-rows body, two for SM90's row-split programs, eight per state head for ``splitv8_tile8``); ``batch`` and ``heads`` must be positive.
    ``tiles`` (prefill): ``min(sm_count, tiles)`` persistent CTAs over the host's ``N x H`` tile count, both
    positive.  The public API refuses an empty call before reaching here."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "tiles":
        if tiles is None or sm_count is None:
            raise ValueError("a 'tiles' grid rule needs tiles and sm_count")
        tiles, sm_count = int(tiles), int(sm_count)
        if tiles <= 0 or sm_count <= 0:
            raise ValueError("tiles and sm_count must be positive")
        return (min(sm_count, tiles), 1, 1)
    batch, heads = int(batch), int(heads)
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")
    if kind != "state_heads":
        raise ValueError(f"unknown Cake GDP grid rule: {kind!r}")
    per_cta = int(rule.get("heads_per_cta", 1))
    return (-(-(batch * heads) // per_cta) * int(rule["ctas_per_state"]), 1, 1)


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake GDP target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake GDP module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    return cake_build.jit_specs(arch)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    batch: int,
    heads: int,
    device_index: int,
    tiles: int | None = None,
    sm_count: int | None = None,
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` (decode) or
    ``PREFILL_LOGICAL_TENSORS`` / ``PREFILL_LOGICAL_SCALARS`` (prefill) to tensors and Python scalars; the module
    record maps its kernel parameter names onto that vocabulary.  The launch grid follows the module's recorded
    grid rule (``launch_grid``: ``batch`` rows of ``heads`` state heads, or ``tiles`` persistent tiles on a device of
    ``sm_count`` SMs) and the kernel runs on the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, batch, heads, tiles=tiles, sm_count=sm_count)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
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
    "F16_STATE_ALIGNMENT_BYTES",
    "HEAD_SIZE_K",
    "HEAD_SIZE_V",
    "INFERENCE_PAD_SPAN",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "NUM_HOUSEHOLDER",
    "PREFILL_FLAGS",
    "PREFILL_LOGICAL_SCALARS",
    "PREFILL_LOGICAL_TENSORS",
    "PREFILL_ROUTE",
    "PREFILL_SPLIT_FLAG",
    "PREFILL_STATE_DTYPES",
    "ROUTES",
    "ROUTE_BODIES",
    "ROUTE_BODY_P3",
    "ROUTE_BODY_V6",
    "STATE_ALIGNMENT_BYTES",
    "STATE_DTYPES",
    "STATE_STRIDE_ELEMENTS",
    "V6Launch",
    "V6Plan",
    "V6Sm90Plan",
    "V6_ARCHES",
    "V6_BODY",
    "V6_PLAN_CANDIDATES",
    "V6_PLAN_COSTS",
    "V6_PLAN_FLOORS",
    "V6_PLAN_MARGIN_PERCENT",
    "V6_PLAN_ROBUST_PERCENT",
    "V6_PLAN_WARMUP_CAP",
    "arch_for_capability",
    "clear_plan_caches",
    "device_arch",
    "inference_csr",
    "inference_lists",
    "inference_placement",
    "inference_row_cost",
    "launch_grid",
    "launch_split",
    "library_name",
    "load",
    "lpt_counts",
    "lpt_place",
    "lpt_rounds",
    "max_num_tokens",
    "prefill_batch_body",
    "prefill_batch_costs",
    "prefill_body_class",
    "prefill_body_classes",
    "prefill_body_costs",
    "prefill_launch_plan",
    "prefill_route_applies",
    "prefill_route_modules",
    "prefill_split_table",
    "prefill_v6_candidate_plan",
    "prefill_v6_launch",
    "prefill_v6_plan",
    "prefill_v6_plan_candidates",
    "prefill_v6_plan_choice",
    "prefill_v6_plan_reference",
    "prefill_v6_plan_scenarios",
    "prefill_v6_record_plan",
    "prefill_v6_sm90_plan",
    "route_applies",
    "route_chunk_steps",
    "route_key",
    "route_modules",
    "run_module",
    "selected_module",
    "selected_prefill_module",
    "selected_v6_module",
    "sm_count",
    "specs_for_arch",
    "split_count",
    "split_table",
    "state_dtype_key",
    "supported_num_householder",
    "v6_householders",
    "v6_launch_of_plan",
    "v6_sm90_item_words",
    "v6_sm90_serial_load",
    "v6_split_bal_slot",
    "v6_split_count",
    "v6_state_form",
    "v6_state_forms",
]
