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
CTA per batch row and head)
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
default).  Every FP16 record is the twin of the FP32 record of the same route,
body, variant and token count, with the same band, and is delivered on SM100 /
SM103 / SM107 only: its stores round the FP32 state with the hardware
``cvt.rs`` instruction.  Its kernel takes two more arguments, the int64 ``rand_seed``
tensor it reads on the device and the Philox round count ``philox_rounds``
(``0`` rounds to nearest even and reads no seed).  The dispatch functions
take the pool's dtype as ``state_dtype`` (``torch.float32`` by default) and
consider the records of that dtype only, so an FP16 pool runs the twin of the
program an FP32 pool would run.

An FP16 call also has a launch class: its round count (``philox_rounds``, 0
without a seed) and whether it stores per-token snapshots
(``ssm_state_indices`` given).  A selection cell of an FP16 record may bound
the round count (``rounds_at_least`` / ``rounds_at_most``, inclusive) and name
the snapshot use (``snapshot``); an absent key admits every value.  Two FP16
programs have a sibling built with ``snapshot_noise``: the ``l8r4`` program of
16 tokens and the runtime program of three Householder updates.  The sibling
draws a launch's random bits once into shared memory instead of at each store,
which pays only when the launch stores snapshots; it keeps its program's body,
variant and band and stores the same bits.  Its cells admit the stochastic
calls with snapshots of the measured cell, and its plain program's cells admit
every other call of the band, so every call still selects exactly one program
and an FP16 pool still runs the body and variant an FP32 pool would run.
The FP16 row-partition program has a sibling built with ``draw_share``: each
Philox draw of its store is computed once and shared by a lane shuffle, so it
stores the same bits.  Its cell admits the stochastic calls of the band
(``rounds_at_least``), and its plain program keeps the calls that round to
nearest even (``rounds_at_most``).

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
count``); ``l8r2``;
``l4r2``; ``l4r4`` for the runtime programs; ``splitv8_tile8`` = eight CTAs of
eight rows
per head), ``num_householder`` (``n``),
``t_steps`` (a fixed token count compiled in; ``None`` for a runtime token
count up to ``max_t_steps``), ``use_qk_l2norm``, on an FP16-pool record
``state_dtype`` and on a sibling ``snapshot_noise`` or ``draw_share``.  Prefill: the six flags
``use_initial_state``, ``use_state_indices``, ``store_final_state``,
``unit_gates``, ``unit_beta`` and ``segment_pass`` beside the defaults of the
kernel's build signature (the I/O dtype ``bf16``, the WY solve arm, the O
store, ...; the split programs spell ``split_items`` too).
"""

from __future__ import annotations

import functools
import heapq
import itertools
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_6bb51453cd15a43da9c2": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_717ce3c684b41c37e673": {
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
        "select_rule": None,
    },
    "cake_gdp_72b6872c98ed9abf0bdf": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_8e26e2c2d0cb0a7fbee9": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
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
    "cake_gdp_b047f2f1a20630e6f36b": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
            "rounds_at_least": 5,
            "snapshot": True,
        },
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
        "select_rule": None,
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
    "cake_gdp_bb55d7d5e4db0d95acce": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_bc76e12e0c247b44a27d": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_c3cf32174475946d9c35": {
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "kind": "state_heads_between",
        },
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_above": 2,
            "kind": "state_heads_between",
        },
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
        "select_rule": {
            "above": 160,
            "at_most": None,
            "ctas_per_sm_above": 4,
            "ctas_per_sm_at_most": 64,
            "kind": "state_heads_between",
        },
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "ctas_per_sm_at_most": 2,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_efdf403e0df822bfc83f": {
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
        "select_rule": None,
    },
    "cake_gdp_f5d305e6f05978caca3c": {
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
                "at_most": None,
                "kind": "state_heads_between",
                "rounds_at_most": 9,
            },
            {
                "above": 0,
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
    "cake_gdp_f108e105bd1ec635650f": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag+tailc",
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
        "kernel": "cake_gdp_f108e105bd1ec635650f",
        "kernel_symbol": "kernel_cake_gdp_f108e105bd1ec635650f",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_f108e105bd1ec635650f",
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
    "cake_gdp_88c5777dabbda5e1e5a8": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag",
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
        "kernel": "cake_gdp_88c5777dabbda5e1e5a8",
        "kernel_symbol": "kernel_cake_gdp_88c5777dabbda5e1e5a8",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_88c5777dabbda5e1e5a8",
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
    "cake_gdp_6f505b4fb9199bf89c96": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag+tailc",
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
        "kernel": "cake_gdp_6f505b4fb9199bf89c96",
        "kernel_symbol": "kernel_cake_gdp_6f505b4fb9199bf89c96",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_6f505b4fb9199bf89c96",
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
    "cake_gdp_1d27e13364ddfad0595c": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag",
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
        "kernel": "cake_gdp_1d27e13364ddfad0595c",
        "kernel_symbol": "kernel_cake_gdp_1d27e13364ddfad0595c",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_1d27e13364ddfad0595c",
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
    "cake_gdp_0ad4bbaf5ef274640c3a": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag+tailc",
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
        "kernel": "cake_gdp_0ad4bbaf5ef274640c3a",
        "kernel_symbol": "kernel_cake_gdp_0ad4bbaf5ef274640c3a",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_0ad4bbaf5ef274640c3a",
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
    "cake_gdp_25d81fdd5cf38d846a0d": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag",
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
        "kernel": "cake_gdp_25d81fdd5cf38d846a0d",
        "kernel_symbol": "kernel_cake_gdp_25d81fdd5cf38d846a0d",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_25d81fdd5cf38d846a0d",
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
    "cake_gdp_2765b8c0ae143504657e": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag+tailc",
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
        "kernel": "cake_gdp_2765b8c0ae143504657e",
        "kernel_symbol": "kernel_cake_gdp_2765b8c0ae143504657e",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_2765b8c0ae143504657e",
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
    "cake_gdp_68d934c41a0fbe0fb027": {
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
            "align": "r2+ctr+tmaw+r3+r3w+qs+c6a+c7+c3+bfold+ord+iss+sa+nomask+bdiag",
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
        "kernel": "cake_gdp_68d934c41a0fbe0fb027",
        "kernel_symbol": "kernel_cake_gdp_68d934c41a0fbe0fb027",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223360,
            "use_pdl": False,
        },
        "library": "cake_gdp_68d934c41a0fbe0fb027",
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
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
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
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
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
            "cake_gdp_f108e105bd1ec635650f",
            "cake_gdp_88c5777dabbda5e1e5a8",
            "cake_gdp_6f505b4fb9199bf89c96",
            "cake_gdp_1d27e13364ddfad0595c",
            "cake_gdp_0ad4bbaf5ef274640c3a",
            "cake_gdp_25d81fdd5cf38d846a0d",
            "cake_gdp_2765b8c0ae143504657e",
            "cake_gdp_68d934c41a0fbe0fb027",
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
    and admits when any cell does.  ``None`` (a runtime program without cells) admits everything."""

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
    if waves_at_most is not None and state_heads > int(waves_at_most) * sm_count:
        return False
    if waves_above is not None and state_heads <= int(waves_above) * sm_count:
        return False
    if waves_at_least is not None and state_heads < int(waves_at_least) * sm_count:
        return False
    if waves_below is not None and state_heads >= int(waves_below) * sm_count:
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
    band of state-head counts (``above < state_heads <= at_most`` within optional wave bounds, multiples of the SM
    count; a band of several intervals is a list of such cells, and a cell may name the ``arches`` it applies to),
    and runtime-token-count programs (``build["t_steps"]`` ``None``, ``T`` up to ``build["max_t_steps"]``) for
    every other call.  Exactly one static program may admit a call; else the admitting runtime program with the
    fewest ``max_t_steps`` runs (two such programs are an error).  Only the records of the pool's dtype take part
    (``build["state_dtype"]``, ``"f32"`` when absent): an FP16 record has the band of its FP32 twin, so both pools
    select the same program body and variant.  The cells of an FP16 record may also bound the launch class
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
    every sequence); None searches every body of the route.  Exactly one program must match."""

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
        spelled = _prefill_flags(MODULES[name]["select_rule"])
        if all(
            spelled.get(flag, False) == value for flag, value in wanted.items()
        ) and not any(value for flag, value in spelled.items() if flag not in wanted):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"{len(matches)} programs of the {PREFILL_ROUTE!r} route on {arch!r} of body {body!r} spell the flags {wanted}: {matches}"
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
# head, cut, span)``.  A call launches one of two builds of its arch: the ``split_items`` build over the chunk-unit
# rows of ``split_table`` (when ``launch_split`` holds: some sequence is cut, or the batch holds an empty sequence,
# whose state copy only the split build runs), else the block build over one block-unit row ``(seq, head, 0, ceil(len
# / 64))`` per non-empty sequence and head, longest first, CTA ``b`` of a grid of ``G`` running rows ``b, b + G, ...``.
# The split build runs per-CTA row lists (``inference_csr``: CTA ``b`` runs rows ``offsets[b] .. offsets[b + 1] - 1``,
# its cut-0 rows first): the long sequences' rows stay where ``split_table`` spreads them and each short row goes to
# a CTA with room under the current maximum modeled load, else to the least-loaded one.  The split rules read each
# sequence alone and a short sequence is one row per head, so short sequences added to a long one never change its
# cuts.  The grid is ``min(SM count, rows of the table)`` and never exceeds the SM count: one CTA fills an SM, so
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


def inference_csr(rows, cu_seqlens, n: int, grid: int, total_tokens: int | None = None):
    """ABI v2.2 (the ``csr`` build): the split_table ``rows`` of ``cu_seqlens`` as per-CTA lists on ``grid`` CTAs,
    returned as the CTA-major table and its ``grid + 1`` offsets (CTA b runs rows ``offsets[b] .. offsets[b + 1] - 1``,
    no pad rows). The long sequences' rows go as in :func:`inference_placement` (row ``at`` of them to CTA ``at %
    grid``, each CTA's cut-0 rows first); then each unsplit sequence's row, longest first, to the least-loaded CTA
    among those whose modeled load plus the row's cost stays within the current maximum (several 1-token rows stack on
    an idle CTA), or, when none has room, to the least-loaded CTA of all."""
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
    lists: list[list[tuple[int, int, int, int]]] = [[] for _ in range(grid)]
    load = [0] * grid
    for at, row in enumerate(r for r in rows if r[0] in split_seqs):
        lists[at % grid].append(row)
        load[at % grid] += inference_row_cost(row, chunks[row[0]], n)
    for b in range(grid):
        lists[b].sort(
            key=lambda r: 0 if r[2] == 0 else 1
        )  # stable: cut-0 rows first, the rounds kept otherwise
    top = max(load, default=0)
    short = sorted(
        (r for r in rows if r[0] not in split_seqs),
        key=lambda r: -inference_row_cost(r, chunks[r[0]], n),
    )
    for row in short:
        cost = inference_row_cost(row, chunks[row[0]], n)
        room = [b for b in range(grid) if load[b] + cost <= top]
        b = min(room or range(grid), key=lambda c: (load[c], c))
        lists[b].append(row)
        load[b] += cost
        top = max(top, load[b])
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
    ``total_tokens`` defaults to the last) on a device of ``sm_count`` SMs (see the rule above)."""

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
    lane-owns-rows body, eight per state head for ``splitv8_tile8``); ``batch`` and ``heads`` must be positive.
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
    "V6Plan",
    "V6Sm90Plan",
    "V6_ARCHES",
    "V6_BODY",
    "arch_for_capability",
    "device_arch",
    "inference_placement",
    "inference_row_cost",
    "launch_grid",
    "launch_split",
    "library_name",
    "load",
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
    "prefill_v6_plan",
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
    "v6_sm90_item_words",
    "v6_sm90_serial_load",
    "v6_state_form",
    "v6_state_forms",
]
