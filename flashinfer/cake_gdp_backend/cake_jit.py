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
``"prefill"``; SM90).

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

Prefill: the ``prefill`` route holds the single-pass programs of two kernel
bodies: the V6 fork-order body (``body`` ``prefill_sm90_v6``: the plain, the
initial-state and the pool-indexed flag sets, final state stored) and the
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
rules of record below always cut).
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
``state_dtype`` and on a sibling ``snapshot_noise``.  Prefill: the six flags
``use_initial_state``, ``use_state_indices``, ``store_final_state``,
``unit_gates``, ``unit_beta`` and ``segment_pass`` beside the defaults of the
kernel's build signature (the I/O dtype ``bf16``, the WY solve arm, the O
store, ...; the split programs spell ``split_items`` too).
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

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
                "kind": "state_heads_between",
                "rounds_at_least": 10,
                "snapshot": True,
                "arches": ["sm_100a", "sm_103a"],
            },
            {
                "above": 0,
                "at_most": None,
                "kind": "state_heads_between",
                "rounds_at_least": 16,
                "snapshot": True,
                "arches": ["sm_107a"],
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
                "arches": ["sm_100a", "sm_103a"],
            },
            {
                "above": 0,
                "at_most": None,
                "kind": "state_heads_between",
                "rounds_at_most": 15,
                "arches": ["sm_107a"],
            },
            {
                "above": 0,
                "at_most": None,
                "kind": "state_heads_between",
                "snapshot": False,
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
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
            "cake_gdp_717ce3c684b41c37e673",
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
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
            "cake_gdp_717ce3c684b41c37e673",
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
    """Whether the verified export owns one prefill call (the single-pass route exists on the device)."""

    arch = arch_for_capability(device_capability)
    if arch is None or int(num_householder) not in NUM_HOUSEHOLDER:
        return False
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

# ---- SPLIT RULES OF RECORD (GDP prefill gate-saturation split, sm_90a; one text in every copy) ----
SPLIT_CHUNK_ROWS = 64  # expanded rows per chunk; chunks come in pairs of 128 rows
SPLIT_MIN_CHUNKS = 24  # a sequence of fewer chunks is one item
SPLIT_MAX = 48  # at most 48 cuts per (sequence, head); the device's forward scan runs at most this many steps
SPLIT_WARMUP_MAX = 8  # a cut's warm-up runs at most 8 chunks from a zero seed
SPLIT_LOG2_THRESHOLD = (
    -24.0
)  # a cut is saturated when the log2 gate decay over its warm-up is <= -24 (2^-24)
SPLIT_SPAN_MIN = (
    (24, 48),
    (12, 24),
    (0, 12),
)  # (H above, least span in chunks): H > 24 -> 48, H > 12 -> 24, else 12
SPLIT_COST_CHUNK = 12  # the cost model, in tenths of a chunk period: per chunk of an item (span + 1 chunks) ...
SPLIT_COST_ITEM = 40  # ... and per item (seed, pipeline fill and drain, final store)
SPLIT_MODES = ("auto", "exact")


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


def split_span_min(H: int) -> int:
    """The least span of an item under ``H`` heads (SPLIT_SPAN_MIN): a wide batch keeps its items long."""
    for h_above, span_min in SPLIT_SPAN_MIN:
        if H > h_above:  # noqa: SIM300
            return span_min
    return SPLIT_SPAN_MIN[-1][1]


def split_span(chunks: int, p: int, n: int) -> int:
    """The span of ``p`` items over ``chunks`` chunks: ``ceil(chunks / p)`` rounded up to the alignment."""
    align = split_align(n)
    return align * (-(-(-(-chunks // p)) // align))  # noqa: B002


def split_count(
    chunks: int, H: int, n: int, sm_count: int, split_mode: str = "auto"
) -> tuple[int, int]:
    """``(p, span)`` of a sequence of ``chunks`` chunks under ``H`` heads with ``n`` householder steps on ``sm_count``
    SMs: the item count and the cut spacing in chunks (cuts at ``j span``, ``j = 1 .. p - 1``; the last item is
    shorter when ``p span > chunks``). ``(1, 0)`` for an empty sequence; ``(1, chunks)`` for ``"exact"`` or below
    SPLIT_MIN_CHUNKS; else, over ``p = 1 .. SPLIT_MAX`` while the span is at least ``split_span_min(H)``, the ``p`` of
    the least cost ``ceil(H p_eff / sm_count) ((span + 1) SPLIT_COST_CHUNK + SPLIT_COST_ITEM)`` (the waves of one
    sequence's ``H p_eff`` items of ``span + 1`` chunks each, with a per-item cost; ``p_eff = ceil(chunks / span)``),
    ties to the smaller ``p``. The rule reads the sequence's own chunk count and the shape constants only, never the
    batch: a sequence splits the same alone and in any batch."""
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode is one of {SPLIT_MODES}, got {split_mode!r}")
    if chunks <= 0:
        return 1, 0
    if split_mode == "exact" or chunks < SPLIT_MIN_CHUNKS:
        return 1, chunks
    span_min = split_span_min(H)
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
) -> list[tuple[int, int, int, int]]:
    """The work items of a batch, the rows of the device's item table (formal 9 of the ``_split`` programs, four int32
    words each; ``total_tiles`` is the row count): per sequence in order, j-major (``j = 0`` of every head, then
    ``j = 1``, ...), ``(seq, head, cut = j span, span)`` for ``(p, span) = split_count(...)`` of the sequence; an empty
    sequence has one item per head (``cut 0, span 0``: the copy of its initial state). ``cu_seqlens`` is any sequence
    of ``N + 1`` integers; ``total_tokens`` defaults to its last entry."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1] if total_tokens is None else int(total_tokens)
    items = []  # type: ignore[var-annotated]
    for seq in range(len(raw) - 1):
        start, end = split_sequence_bounds(raw[seq], raw[seq + 1], total)
        p, span = split_count(split_chunks(end - start, n), H, n, sm_count, split_mode)
        items.extend((seq, head, j * span, span) for j in range(p) for head in range(H))
    return items


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
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "NUM_HOUSEHOLDER",
    "PREFILL_FLAGS",
    "PREFILL_LOGICAL_SCALARS",
    "PREFILL_LOGICAL_TENSORS",
    "PREFILL_ROUTE",
    "PREFILL_SPLIT_FLAG",
    "ROUTES",
    "ROUTE_BODIES",
    "ROUTE_BODY_P3",
    "ROUTE_BODY_V6",
    "STATE_ALIGNMENT_BYTES",
    "STATE_DTYPES",
    "STATE_STRIDE_ELEMENTS",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "max_num_tokens",
    "prefill_body_class",
    "prefill_body_classes",
    "prefill_body_costs",
    "prefill_launch_plan",
    "prefill_route_applies",
    "prefill_route_modules",
    "prefill_split_table",
    "route_applies",
    "route_chunk_steps",
    "route_key",
    "route_modules",
    "run_module",
    "selected_module",
    "selected_prefill_module",
    "sm_count",
    "specs_for_arch",
    "split_count",
    "split_table",
    "state_dtype_key",
    "supported_num_householder",
]
