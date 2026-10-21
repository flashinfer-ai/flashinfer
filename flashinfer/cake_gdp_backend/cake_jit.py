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

Host plan and loader glue of the Cake Gated DeltaProduct (GDP) decode kernels
for Nemotron 4 (``K = 128``, ``V = 64``, ``n <= 4`` Householder updates per
token, FP32 state pool; SM100 / SM103).

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

A route is the list of module records one public call may launch:
``decode_nh<n>`` for ``n`` Householder updates per token with the q / k L2
normalization fused in the kernel.  One record is launched per call: the
static-token-count program of the call's ``T`` whose band admits the call's
``B x HV`` state heads on the device's SM count (``selected_module``), else the
runtime-token-count program of the route (``T`` up to its ``max_t_steps``).
Batch size and head count are runtime scalars, so a changed problem size never
recompiles anything.  Two kernel bodies ship: the lane-owns-rows body
(``variant`` ``l8r4`` / ``l8r2``, one CTA per batch row and head) and, for
``T = 1`` with ``B x HV <= 160``, the row-partition body (``splitv8_tile8``,
eight CTAs per state head); each record names its body (``body``: ``lanerow`` for
the lane-owns-rows body, ``splitv`` for the row partition).

The program name ``cake_gdp_<20 hex>`` is the first twenty hex digits of the
SHA-256 of the compact JSON ``{"build": <build keywords, sorted>, "module":
<dotted kernel module>}`` of the kernel the program was rendered from; the
record's ``build`` spells those keywords: ``variant`` (the body's geometry:
``l8r4`` = eight lanes per state row, four rows per lane; ``l8r2``; ``l4r4``
for the runtime programs; ``splitv8_tile8`` = eight CTAs of eight rows per
head), ``num_householder`` (``n``),
``t_steps`` (a fixed token count compiled in; ``None`` for a runtime token
count up to ``max_t_steps``) and ``use_qk_l2norm``.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_gdp_27eb88658ee4bdc4ae35": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
        "select_rule": {
            "above": 0,
            "at_most": None,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_3342c82179fa758b0570": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "sm_100a",
            "sm_103a",
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
            "sm_100a",
            "sm_103a",
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
    "cake_gdp_4d40c98af90453032200": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "ctas_per_sm_above": 4,
            "ctas_per_sm_at_most": 5,
            "kind": "state_heads_between",
        },
    },
    "cake_gdp_585a5ea4cbdd07f1b2a8": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "sm_100a",
            "sm_103a",
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
    "cake_gdp_78fc9caafee16e2af047": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
    "cake_gdp_f2f0c7abbf10082a6996": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "dynamic_smem_bytes": 6400,
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
    "cake_gdp_fefa1931e0ddadb74891": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
                "at_most": None,
                "ctas_per_sm_at_most": 4,
                "kind": "state_heads_between",
            },
            {
                "above": 160,
                "at_most": None,
                "ctas_per_sm_above": 5,
                "kind": "state_heads_between",
            },
        ],
    },
}
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "decode_nh1": [
            "cake_gdp_78fc9caafee16e2af047",
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
        ],
        "decode_nh3": [
            "cake_gdp_f2f0c7abbf10082a6996",
            "cake_gdp_fefa1931e0ddadb74891",
            "cake_gdp_4d40c98af90453032200",
            "cake_gdp_27eb88658ee4bdc4ae35",
            "cake_gdp_3cbdef34ce5b580c2216",
            "cake_gdp_5970bfa6676087a3ce92",
            "cake_gdp_585a5ea4cbdd07f1b2a8",
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
        ],
    },
    "sm_103a": {
        "decode_nh1": [
            "cake_gdp_78fc9caafee16e2af047",
        ],
        "decode_nh2": [
            "cake_gdp_3342c82179fa758b0570",
        ],
        "decode_nh3": [
            "cake_gdp_f2f0c7abbf10082a6996",
            "cake_gdp_fefa1931e0ddadb74891",
            "cake_gdp_4d40c98af90453032200",
            "cake_gdp_27eb88658ee4bdc4ae35",
            "cake_gdp_3cbdef34ce5b580c2216",
            "cake_gdp_5970bfa6676087a3ce92",
            "cake_gdp_585a5ea4cbdd07f1b2a8",
        ],
        "decode_nh4": [
            "cake_gdp_3425d6511e4eab4896cd",
        ],
    },
}

# One verified build per architecture: B200 / GB200 (``sm_100a``) and B300 / GB300
# (``sm_103a``).  Every architecture is compiled with its exact ``-gencode`` flag
# set (``build.json`` ``arch_flags``); there is no family fallback.
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
HEAD_SIZE_K = 128
HEAD_SIZE_V = 64
NUM_HOUSEHOLDER = (1, 2, 3, 4)
# The FP32 state pool is read with 16-byte asynchronous copies: a 16-byte aligned
# base and strides that are multiples of four elements.  One-line switch: a
# 256-bit-store program raises these to 32 bytes / eight elements.
STATE_ALIGNMENT_BYTES = 16
STATE_STRIDE_ELEMENTS = 4
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
)
LOGICAL_SCALARS = (
    "heads",
    "num_tokens",
    "scale",
    "disable_state_update",
    "use_ssm_state_indices",
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


def supported_num_householder(arch: str, use_qk_l2norm: bool = True) -> tuple[int, ...]:
    """The ``n`` values with an exported route on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(n for n in NUM_HOUSEHOLDER if route_key(n, use_qk_l2norm) in routes)


def max_num_tokens(arch: str, num_householder: int, use_qk_l2norm: bool = True) -> int:
    """The largest token count ``T`` an exported program of the route admits (0 when the route has none)."""

    key = route_key(num_householder, use_qk_l2norm)
    names = ROUTES.get(arch, {}).get(key, ())
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
) -> bool:
    """Whether the verified export owns one decode call."""

    arch = arch_for_capability(device_capability)
    if arch is None or int(num_householder) not in NUM_HOUSEHOLDER:
        return False
    return 1 <= int(num_tokens) <= max_num_tokens(arch, num_householder, use_qk_l2norm)


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


def _selects(rule: Any, state_heads: int, sm_count: int) -> bool:
    """Whether a static program's band admits ``state_heads = B x HV`` on a device of ``sm_count`` SMs:
    ``above < state_heads <= at_most`` (``at_most`` ``None`` = unbounded) within the wave bounds
    ``state_heads <= ctas_per_sm_at_most x sm_count`` / ``state_heads > ctas_per_sm_above x sm_count`` when the
    rule carries them.  A list of such cells is a band of several intervals and admits when any cell does.
    ``None`` (a runtime program) admits everything."""

    if rule is None:
        return True
    if isinstance(rule, list):
        return any(_selects(cell, state_heads, sm_count) for cell in rule)
    kind = rule["kind"]
    if kind != "state_heads_between":
        raise ValueError(f"unknown Cake GDP decode selection rule: {kind!r}")
    state_heads, sm_count = int(state_heads), int(sm_count)
    at_most = rule["at_most"]
    if state_heads <= int(rule["above"]) or (
        at_most is not None and state_heads > int(at_most)
    ):
        return False
    waves_at_most, waves_above = (
        rule.get("ctas_per_sm_at_most"),
        rule.get("ctas_per_sm_above"),
    )
    if waves_at_most is not None and state_heads > int(waves_at_most) * sm_count:
        return False
    if waves_above is not None and state_heads <= int(waves_above) * sm_count:
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
) -> str:
    """The one module a public call launches on ``arch`` for ``num_tokens`` tokens of ``state_heads = B x HV``
    state heads on a device of ``sm_count`` SMs (``sm_count(device_index)``).

    This is the band rule of the kernel (its ``band_variant``), spelled once as the records' ``select_rule``
    leaves: a route carries static-token-count programs (``build["t_steps"]`` = the call's ``T``), each owning a
    band of state-head counts (``above < state_heads <= at_most`` within optional wave bounds, multiples of the SM
    count; a band of several intervals is a list of such cells), and one runtime-token-count program (``build["t_steps"]`` ``None``, ``T`` up to
    ``build["max_t_steps"]``) for every other call.  Exactly one static program may admit a call; the runtime
    program is the fallback.
    """

    num_tokens, state_heads = int(num_tokens), int(state_heads)
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if int(sm_count) < 1:
        raise ValueError("sm_count must be positive")
    names = route_modules(arch, num_householder, use_qk_l2norm)
    static = [
        name
        for name in names
        if MODULES[name]["build"]["t_steps"] == num_tokens
        and _selects(MODULES[name]["select_rule"], state_heads, sm_count)
    ]
    if len(static) > 1:
        raise ValueError(
            f"{len(static)} static programs of the {route_key(num_householder, use_qk_l2norm)!r} route on {arch!r} "
            f"admit T={num_tokens}, {state_heads} state heads: {static}"
        )
    if static:
        return static[0]
    runtime = sorted(
        (
            name
            for name in names
            if MODULES[name]["build"]["t_steps"] is None
            and MODULES[name]["build"]["max_t_steps"] >= num_tokens
        ),
        key=lambda name: MODULES[name]["build"]["max_t_steps"],
    )
    if not runtime:
        raise ValueError(
            f"no exported module of the {route_key(num_householder, use_qk_l2norm)!r} route on {arch!r} admits "
            f"T={num_tokens} tokens (at most {max_num_tokens(arch, num_householder, use_qk_l2norm)})"
        )
    return runtime[0]


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(record: dict[str, Any], batch: int, heads: int) -> tuple[int, int, int]:
    """Launch grid of one module under the recorded ``state_heads`` grid rule: ``ceil(B x HV / heads_per_cta) x
    ctas_per_state`` CTAs (``heads_per_cta`` defaults to one: one CTA per batch row and head for the lane-owns-rows
    body, eight per state head for ``splitv8_tile8``).  ``batch`` and ``heads`` must be positive; the public API
    refuses an empty call before reaching here."""

    batch, heads = int(batch), int(heads)
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")
    rule = record["grid_rule"]
    if rule["kind"] != "state_heads":
        raise ValueError(f"unknown Cake GDP decode grid rule: {rule['kind']!r}")
    per_cta = int(rule.get("heads_per_cta", 1))
    return (-(-(batch * heads) // per_cta) * int(rule["ctas_per_state"]), 1, 1)


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake GDP decode target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake GDP decode module {name} is not delivered for {arch}; its architectures: {record['arches']}"
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
    name: str, values: dict[str, Any], *, batch: int, heads: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to tensors and Python scalars;
    the module record maps its kernel parameter names onto that vocabulary.  The launch grid follows the module's
    recorded grid rule for ``batch`` rows of ``heads`` state heads and the kernel runs on the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, batch, heads)
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
    "HEAD_SIZE_K",
    "HEAD_SIZE_V",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "NUM_HOUSEHOLDER",
    "ROUTES",
    "STATE_ALIGNMENT_BYTES",
    "STATE_STRIDE_ELEMENTS",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "max_num_tokens",
    "route_applies",
    "route_key",
    "route_modules",
    "run_module",
    "selected_module",
    "sm_count",
    "specs_for_arch",
    "supported_num_householder",
]
