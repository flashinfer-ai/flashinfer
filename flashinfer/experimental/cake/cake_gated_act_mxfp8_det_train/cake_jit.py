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

Host plan and loader glue of the Cake fused SwiGLU + MXFP8 quantization
training kernels (forward and backward; SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch
record, grid rule, selection rule and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is one direction and one output set: ``fwd_row``, ``fwd_col``,
``fwd_both``, ``bwd_row``, ``bwd_col`` and ``bwd_both`` (:func:`route_name`).
The forward reads ``gated_input`` ``[M, 2K]`` (gate columns, then up columns)
and quantizes ``silu(gate) * up`` (``[M, K]``); the backward also reads
``grad_output`` ``[M, K]`` and quantizes the gate and up gradients
(``[M, 2K]``).  Each route lists every program a call may run and runs
exactly one, chosen by the problem size: a record's ``select_rule``
(``elements_at_least``: ``M * K >= elements[arch]``) names the sizes it owns,
of several owners the one of the highest threshold runs, and the record
without a threshold on the architecture is the route's fallback there.
:func:`selected_module` applies that rule, a pure function of the
architecture, the route and ``M * K``; the host never reads the data, and
every program of a route writes the same bytes.

A program launches one CTA per tile: ``tile_rows`` rows by ``tile_cols``
columns of the gate half (:func:`launch_grid`): 128 columns for the forward and
64 for the backward, or a narrower tile (64 or 32 columns for the forward, 32
for the backward) that spreads a small problem over more SMs, and 32 rows, or
64 or 128 for a program whose threads own two or four rows each, or 8 for a
program that spends 16 lanes on each row.  ``M`` and
``K`` are runtime scalars, so a changed problem size never recompiles anything;
building every library of the architecture (:func:`specs_for_arch`) before a
CUDA Graph capture keeps the JIT out of the capture.

The program name ``cake_gated_act_mxfp8_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
sorted>, "module": <dotted kernel module>}`` of the kernel the program was
rendered from; each record's comment names its build.
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
    # The forward, rowwise, with an 8 x 32 tile of 16 lanes per row, the
    # branchless exact sigmoid and the redux block maximum: the route's
    # fallback on sm_100a, sm_103a and sm_107a (no select_rule).
    "cake_gated_act_mxfp8_a097ced99ffe2f27b973": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 8,
        },
        "kernel": "cake_gated_act_mxfp8_a097ced99ffe2f27b973",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_a097ced99ffe2f27b973",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_a097ced99ffe2f27b973",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": None,
    },
    # The forward, rowwise, with a 32 x 32 tile and the hybrid band path (on
    # sm_100a and sm_107a only): it runs when M * K >= 2^15 (2^17 on sm_107a)
    # and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_9e34b70ff5e2423e9a42": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_9e34b70ff5e2423e9a42",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_9e34b70ff5e2423e9a42",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_9e34b70ff5e2423e9a42",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 32768,
                "sm_107a": 131072,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with 4 CTAs per SM and the inline band path (on
    # sm_107a only): it runs when M * K >= 2^19 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_7711f8af92d5b9756501": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_7711f8af92d5b9756501",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_7711f8af92d5b9756501",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_7711f8af92d5b9756501",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 524288,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with 8 CTAs per SM, the packed sigmoid and the
    # inline band path (on sm_100a only): it runs when M * K >= 2^22 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_7ed7c74a790b9b536276": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_7ed7c74a790b9b536276",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_7ed7c74a790b9b536276",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_7ed7c74a790b9b536276",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with 4 CTAs per SM, the packed sigmoid and the
    # inline band path (on sm_100a only): it runs when M * K >= 2^20 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_fd6057ad400d80a36db9": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_fd6057ad400d80a36db9",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_fd6057ad400d80a36db9",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_fd6057ad400d80a36db9",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with no_allocate loads, 8 CTAs per SM, the packed
    # sigmoid and the inline band path (on sm_100a only): it runs when M * K >=
    # 2^28 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_baf6fbeb97918f613cef": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_baf6fbeb97918f613cef",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_baf6fbeb97918f613cef",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_baf6fbeb97918f613cef",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 268435456,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with 6 CTAs per SM, the packed sigmoid and the ffs
    # band path (on sm_103a only): it runs when M * K >= 5 x 2^19 and no record
    # of a higher threshold admits the call.
    "cake_gated_act_mxfp8_03a64d181c25359b04fd": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_03a64d181c25359b04fd",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_03a64d181c25359b04fd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_03a64d181c25359b04fd",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 2621440,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with 4 CTAs per SM, the packed sigmoid and the ffs
    # band path (on sm_103a only): it runs when M * K >= 2^20 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_e399747ca59173122fb7": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_e399747ca59173122fb7",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_e399747ca59173122fb7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_e399747ca59173122fb7",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with no_allocate loads, 8 CTAs per SM and the
    # inline band path (on sm_103a only): it runs when M * K >= 2^28 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_69fc759322d18bf4aa70": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_69fc759322d18bf4aa70",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_69fc759322d18bf4aa70",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_69fc759322d18bf4aa70",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 268435456,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with two rows per thread, 4 CTAs per SM, the packed
    # sigmoid and the ffs band path (on sm_107a only): it runs when M * K >=
    # 2^23 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_9e0e07c8dd55b72b2430": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_9e0e07c8dd55b72b2430",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_9e0e07c8dd55b72b2430",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_9e0e07c8dd55b72b2430",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 8388608,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with four rows per thread and 3 CTAs per SM (on
    # sm_107a only): it runs when M * K >= 2^26 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_81973835902ebc40b9b5": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 128,
        },
        "kernel": "cake_gated_act_mxfp8_81973835902ebc40b9b5",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_81973835902ebc40b9b5",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_81973835902ebc40b9b5",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with four rows per thread, no_allocate loads and 3
    # CTAs per SM (on sm_107a only): it runs when M * K >= 2^29 and no record
    # of a higher threshold admits the call.
    "cake_gated_act_mxfp8_d482e0e2dc5d60ddeb62": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 128,
        },
        "kernel": "cake_gated_act_mxfp8_d482e0e2dc5d60ddeb62",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_d482e0e2dc5d60ddeb62",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_d482e0e2dc5d60ddeb62",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 536870912,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with 6 CTAs per SM, the packed sigmoid and the
    # inline band path (on sm_100a only): it runs when M * K >= 5 x 2^19 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_a95dec432abb6b7d2109": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_a95dec432abb6b7d2109",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_a95dec432abb6b7d2109",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_a95dec432abb6b7d2109",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 2621440,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with two rows per thread, 4 CTAs per SM and the
    # inline band path (on sm_107a only): it runs when M * K >= 5 x 2^20 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_83485c623aae181a44f7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_83485c623aae181a44f7",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_83485c623aae181a44f7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_83485c623aae181a44f7",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 5242880,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with a 32 x 32 tile, 6 CTAs per SM and the hybrid
    # band path (on sm_103a only): it runs when M * K >= 2^18 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_40b0da5244b08b1db510": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_40b0da5244b08b1db510",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_40b0da5244b08b1db510",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_40b0da5244b08b1db510",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 262144,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 32 tile and the hybrid band path: the
    # route's fallback on sm_107a (no threshold there); on sm_103a it runs when
    # M * K >= 2^20 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_7c3f4b291326f333e943": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_7c3f4b291326f333e943",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_7c3f4b291326f333e943",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_7c3f4b291326f333e943",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_103a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 32 tile, 8 CTAs per SM and the hybrid
    # band path: the route's fallback on sm_100a and sm_103a (no select_rule).
    "cake_gated_act_mxfp8_33c80e05bad314060124": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_33c80e05bad314060124",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_33c80e05bad314060124",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_33c80e05bad314060124",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": None,
    },
    # The forward, colwise, with 8 CTAs per SM, the packed sigmoid and the
    # called band path (on sm_100a only): it runs when M * K >= 2^22 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6ae0a2c005d5fc854efc": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_6ae0a2c005d5fc854efc",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6ae0a2c005d5fc854efc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6ae0a2c005d5fc854efc",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread, 5 CTAs per SM, the packed
    # sigmoid and the called band path (on sm_100a only): it runs when M * K >=
    # 2^24 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_039a5afd9072a9295001": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_039a5afd9072a9295001",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_039a5afd9072a9295001",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_039a5afd9072a9295001",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread, no_allocate loads, 5 CTAs
    # per SM and the inline band path (on sm_100a only): it runs when M * K >=
    # 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread, no_allocate loads, 5 CTAs
    # per SM, the packed sigmoid and the called band path (on sm_100a and
    # sm_103a only): it runs when M * K >= 2^27 (2^26 on sm_103a) and no record
    # of a higher threshold admits the call.
    "cake_gated_act_mxfp8_b7975e9eda41aefb0456": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 134217728,
                "sm_103a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 64 tile (on sm_103a only): it runs when
    # M * K >= 2^22 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_f1232b8d8f9b3a7f66cf": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_f1232b8d8f9b3a7f66cf",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_f1232b8d8f9b3a7f66cf",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_f1232b8d8f9b3a7f66cf",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_103a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 64 tile and the packed sigmoid (on
    # sm_103a only): it runs when M * K >= 2^23 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_363476aea39117786ffc": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_363476aea39117786ffc",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_363476aea39117786ffc",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_363476aea39117786ffc",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_103a": 8388608,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread, 5 CTAs per SM, the packed
    # sigmoid and the ffs band path (on sm_103a only): it runs when M * K >=
    # 2^24 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_7b36d1a18f26d26d8692": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_7b36d1a18f26d26d8692",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_7b36d1a18f26d26d8692",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_7b36d1a18f26d26d8692",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_103a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 64 tile and the ffs band path (on
    # sm_107a only): it runs when M * K >= 2^20 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_0827951f83a4f724f2a0": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_0827951f83a4f724f2a0",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_0827951f83a4f724f2a0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_0827951f83a4f724f2a0",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread and the called band path
    # (on sm_107a only): it runs when M * K >= 2^22 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_96497ddc9daf22dc35a5": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_96497ddc9daf22dc35a5",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_96497ddc9daf22dc35a5",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_96497ddc9daf22dc35a5",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with four rows per thread, no_allocate loads, 3
    # CTAs per SM and the packed sigmoid (on sm_107a only): it runs when M * K
    # >= 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6902c8391e6f660586a4": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 128,
        },
        "kernel": "cake_gated_act_mxfp8_6902c8391e6f660586a4",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6902c8391e6f660586a4",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 32768,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6902c8391e6f660586a4",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread, the packed sigmoid and
    # the inline band path (on sm_107a only): it runs when M * K >= 2^24 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_2e3128bb6b187c7a4a7c": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_2e3128bb6b187c7a4a7c",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_2e3128bb6b187c7a4a7c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_2e3128bb6b187c7a4a7c",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 32 tile, the packed sigmoid and the
    # hybrid band path (on sm_100a only): it runs when M * K >= 2^20 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6ffdb2bd1b07f3dbf991": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_6ffdb2bd1b07f3dbf991",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6ffdb2bd1b07f3dbf991",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6ffdb2bd1b07f3dbf991",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with a 32 x 32 tile, 8 CTAs per SM and
    # the ffs band path: the route's fallback on sm_100a (no select_rule).
    "cake_gated_act_mxfp8_5e2d60e2f8ccce9e3ceb": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_5e2d60e2f8ccce9e3ceb",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_5e2d60e2f8ccce9e3ceb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_5e2d60e2f8ccce9e3ceb",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": None,
    },
    # The forward, rowwise and colwise, with a 32 x 32 tile and the hybrid band
    # path: the route's fallback on sm_107a (no select_rule).
    "cake_gated_act_mxfp8_df1c4b7c00309eb06349": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_df1c4b7c00309eb06349",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_df1c4b7c00309eb06349",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_df1c4b7c00309eb06349",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": None,
    },
    # The forward, rowwise and colwise, with a 32 x 32 tile, 8 CTAs per SM and
    # the hybrid band path: the route's fallback on sm_103a (no select_rule).
    "cake_gated_act_mxfp8_480a2547e3b6273df280": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_480a2547e3b6273df280",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_480a2547e3b6273df280",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_480a2547e3b6273df280",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": None,
    },
    # The forward, rowwise and colwise, with 8 CTAs per SM, the packed sigmoid
    # and the called band path (on sm_100a only): it runs when M * K >= 2^22
    # and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_f536b838e9f5b329b548": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_f536b838e9f5b329b548",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_f536b838e9f5b329b548",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_f536b838e9f5b329b548",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, 4 CTAs per SM, the packed sigmoid and the ffs band path (on
    # sm_100a only): it runs when M * K >= 2^24 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_926bcb202902eef2465c": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_926bcb202902eef2465c",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_926bcb202902eef2465c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_926bcb202902eef2465c",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise (on sm_107a only): it runs when M * K >=
    # 3 x 2^20 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_0520c14fde53ef7bfea7": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_0520c14fde53ef7bfea7",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_0520c14fde53ef7bfea7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_0520c14fde53ef7bfea7",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 3145728,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with the called band path (on sm_107a
    # only): it runs when M * K >= 2^20 and no record of a higher threshold
    # admits the call.
    "cake_gated_act_mxfp8_d4842157b309640e4317": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_d4842157b309640e4317",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_d4842157b309640e4317",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_d4842157b309640e4317",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with 8 CTAs per SM and the packed
    # sigmoid (on sm_103a only): it runs when M * K >= 2^21 and no record of a
    # higher threshold admits the call.
    "cake_gated_act_mxfp8_03452b1f2ae3d40b1bcc": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_03452b1f2ae3d40b1bcc",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_03452b1f2ae3d40b1bcc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_03452b1f2ae3d40b1bcc",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_103a": 2097152,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with two rows per thread, no_allocate
    # loads and the packed sigmoid (on sm_103a only): it runs when M * K >=
    # 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_31deb882beba0de9d64a": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_31deb882beba0de9d64a",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_31deb882beba0de9d64a",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_31deb882beba0de9d64a",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_103a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with four rows per thread, no_allocate
    # loads, 3 CTAs per SM, the packed sigmoid and the called band path (on
    # sm_103a only): it runs when M * K >= 7 x 2^24 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_ba8d72750dad06fadf5d": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 128,
        },
        "kernel": "cake_gated_act_mxfp8_ba8d72750dad06fadf5d",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_ba8d72750dad06fadf5d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 32768,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_ba8d72750dad06fadf5d",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_103a": 117440512,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, the packed sigmoid and the called band path (on sm_107a only): it
    # runs when M * K >= 2^23 and no record of a higher threshold admits the
    # call.
    "cake_gated_act_mxfp8_c116ff68e21df5068f6d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_c116ff68e21df5068f6d",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c116ff68e21df5068f6d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c116ff68e21df5068f6d",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 8388608,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with four rows per thread, no_allocate
    # loads, 3 CTAs per SM and the packed sigmoid (on sm_107a only): it runs
    # when M * K >= 3 x 2^23 and no record of a higher threshold admits the
    # call.
    "cake_gated_act_mxfp8_a133fc612a9b9a21e84b": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 128,
            "tile_rows": 128,
        },
        "kernel": "cake_gated_act_mxfp8_a133fc612a9b9a21e84b",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_a133fc612a9b9a21e84b",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 32768,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_a133fc612a9b9a21e84b",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 25165824,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with a 32 x 32 tile, the packed sigmoid
    # and the ffs band path (on sm_100a only): it runs when M * K >= 2^20 and
    # no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_d1729b9c66e203234b04": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_d1729b9c66e203234b04",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_d1729b9c66e203234b04",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_d1729b9c66e203234b04",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "fwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with an 8 x 32 tile of 16 lanes per row, the
    # branchless exact sigmoid and the redux block maximum: the route's
    # fallback on sm_100a, sm_103a and sm_107a (no select_rule).
    "cake_gated_act_mxfp8_6d93c6436dd21ee79954": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 8,
        },
        "kernel": "cake_gated_act_mxfp8_6d93c6436dd21ee79954",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6d93c6436dd21ee79954",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6d93c6436dd21ee79954",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": None,
    },
    # The backward, rowwise, with a 32 x 32 tile, 12 CTAs per SM and the called
    # band path (on sm_100a only): it runs when M * K >= 2^18 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_9c68fc985d41586ecbe8": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_9c68fc985d41586ecbe8",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_9c68fc985d41586ecbe8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_9c68fc985d41586ecbe8",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 262144,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with a 32 x 32 tile, 12 CTAs per SM and the inline
    # band path (on sm_103a only): it runs when M * K >= 2^18 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_54573bbcd9b4aadbd39f": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_54573bbcd9b4aadbd39f",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_54573bbcd9b4aadbd39f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_54573bbcd9b4aadbd39f",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 262144,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with a 32 x 32 tile (on sm_107a only): it runs
    # when M * K >= 2^17 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_8da8bc14e105ce409f74": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_8da8bc14e105ce409f74",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_8da8bc14e105ce409f74",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_8da8bc14e105ce409f74",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 131072,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with 9 CTAs per SM, the packed sigmoid and the
    # called band path (on sm_100a only): it runs when M * K >= 2^20 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_5bf054fcafaa9784cea3": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_5bf054fcafaa9784cea3",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_5bf054fcafaa9784cea3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_5bf054fcafaa9784cea3",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with no_allocate loads and the inline band path
    # (on sm_100a only): it runs when M * K >= 2^26 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_91103eb79c29acb201d7": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_91103eb79c29acb201d7",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_91103eb79c29acb201d7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_91103eb79c29acb201d7",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with two rows per thread, no_allocate loads, 7
    # CTAs per SM, the packed sigmoid and the called band path (on sm_100a
    # only): it runs when M * K >= 2^28 and no record of a higher threshold
    # admits the call.
    "cake_gated_act_mxfp8_bdbe30b9562eca355986": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_bdbe30b9562eca355986",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_bdbe30b9562eca355986",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_bdbe30b9562eca355986",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_100a": 268435456,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with 9 CTAs per SM, the packed sigmoid and the ffs
    # band path (on sm_103a only): it runs when M * K >= 2^20 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_8d0ef9eac275dd44006c": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_8d0ef9eac275dd44006c",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_8d0ef9eac275dd44006c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_8d0ef9eac275dd44006c",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with 9 CTAs per SM, the packed sigmoid and the
    # inline band path (on sm_103a only): it runs when M * K >= 2^24 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_1f02c63ba6b7d1e0a3af": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_1f02c63ba6b7d1e0a3af",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_1f02c63ba6b7d1e0a3af",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_1f02c63ba6b7d1e0a3af",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with no_allocate loads, 10 CTAs per SM, the packed
    # sigmoid and the called band path (on sm_103a only): it runs when M * K >=
    # 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6d1f382d999ef2f444c7": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_6d1f382d999ef2f444c7",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6d1f382d999ef2f444c7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6d1f382d999ef2f444c7",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_103a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with 8 CTAs per SM, the packed sigmoid and the
    # called band path (on sm_107a only): it runs when M * K >= 2^18 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_f8db01d14808a5afc6da": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_f8db01d14808a5afc6da",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_f8db01d14808a5afc6da",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_f8db01d14808a5afc6da",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 262144,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with two rows per thread, no_allocate loads, 8
    # CTAs per SM and the ffs band path (on sm_107a only): it runs when M * K
    # >= 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_c26c9ea40f45054678ac": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_c26c9ea40f45054678ac",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c26c9ea40f45054678ac",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c26c9ea40f45054678ac",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise, with two rows per thread, no_allocate loads and
    # the inline band path (on sm_107a only): it runs when M * K >= 7 x 2^23
    # and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_dc95c4d586c964293ee9": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_dc95c4d586c964293ee9",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_dc95c4d586c964293ee9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_dc95c4d586c964293ee9",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "row",
        "select_rule": {
            "elements": {
                "sm_107a": 58720256,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with a 32 x 32 tile, 12 CTAs per SM and the hybrid
    # band path: the route's fallback on sm_100a (no select_rule).
    "cake_gated_act_mxfp8_4a560a1480cd1898ba6c": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_4a560a1480cd1898ba6c",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_4a560a1480cd1898ba6c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_4a560a1480cd1898ba6c",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": None,
    },
    # The backward, colwise, with a 32 x 32 tile, 6 CTAs per SM and the hybrid
    # band path: the route's fallback on sm_103a (no select_rule).
    "cake_gated_act_mxfp8_ede2112840c4b6e1ffff": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_ede2112840c4b6e1ffff",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_ede2112840c4b6e1ffff",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_ede2112840c4b6e1ffff",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": None,
    },
    # The backward, colwise, with a 32 x 32 tile, 9 CTAs per SM and the inline
    # band path: the route's fallback on sm_107a (no select_rule).
    "cake_gated_act_mxfp8_c6ab778cd390eb44c75a": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_c6ab778cd390eb44c75a",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c6ab778cd390eb44c75a",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c6ab778cd390eb44c75a",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": None,
    },
    # The backward, colwise, with the packed sigmoid (on sm_100a only): it runs
    # when M * K >= 2^21 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_848a93885f596fa4cbf9": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_848a93885f596fa4cbf9",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_848a93885f596fa4cbf9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_848a93885f596fa4cbf9",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 2097152,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with a 32 x 32 tile, the packed sigmoid and the
    # hybrid band path (on sm_107a only): it runs when M * K >= 2^19 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_78eaec48e0704cf3bea2": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_78eaec48e0704cf3bea2",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_78eaec48e0704cf3bea2",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_78eaec48e0704cf3bea2",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 524288,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with the packed sigmoid and the called band path
    # (on sm_103a only): it runs when M * K >= 2^21 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_addcbe7051b411a1e1e0": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_addcbe7051b411a1e1e0",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_addcbe7051b411a1e1e0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_addcbe7051b411a1e1e0",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_103a": 2097152,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with 10 CTAs per SM, the packed sigmoid and the
    # called band path (on sm_100a only): it runs when M * K >= 2^22 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_5fe5136799b53b96cfc6": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_5fe5136799b53b96cfc6",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_5fe5136799b53b96cfc6",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_5fe5136799b53b96cfc6",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with two rows per thread, no_allocate loads, 6
    # CTAs per SM, the packed sigmoid and the called band path (on sm_100a and
    # sm_103a only): it runs when M * K >= 2^26 (2^24 on sm_103a) and no record
    # of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 67108864,
                "sm_103a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with two rows per thread, no_allocate loads, the
    # packed sigmoid and the hybrid band path (on sm_100a only): it runs when M
    # * K >= 2^24 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_418be6031aeee9031cfe": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_418be6031aeee9031cfe",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_418be6031aeee9031cfe",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_418be6031aeee9031cfe",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_100a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with two rows per thread, no_allocate loads, 12
    # CTAs per SM, the packed sigmoid and the called band path (on sm_107a
    # only): it runs when M * K >= 2^24 and no record of a higher threshold
    # admits the call.
    "cake_gated_act_mxfp8_ca9917264bb212d23139": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_ca9917264bb212d23139",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_ca9917264bb212d23139",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_ca9917264bb212d23139",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with a 32 x 32 tile, 8 CTAs per SM and the called
    # band path (on sm_103a only): it runs when M * K >= 2^15 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_c2ea56613f8ca922571a": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_c2ea56613f8ca922571a",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c2ea56613f8ca922571a",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c2ea56613f8ca922571a",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_103a": 32768,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with 8 CTAs per SM, the packed sigmoid and the
    # hybrid band path (on sm_107a only): it runs when M * K >= 2^23 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_b1cc52ca3532a49c981e": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_b1cc52ca3532a49c981e",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_b1cc52ca3532a49c981e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_b1cc52ca3532a49c981e",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 8388608,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with four rows per thread, no_allocate loads and 4
    # CTAs per SM (on sm_107a only): it runs when M * K >= 2^28 and no record
    # of a higher threshold admits the call.
    "cake_gated_act_mxfp8_49303287790d1f318420": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 128,
        },
        "kernel": "cake_gated_act_mxfp8_49303287790d1f318420",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_49303287790d1f318420",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 32768,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_49303287790d1f318420",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 268435456,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with the packed sigmoid and the ffs band path (on
    # sm_107a only): it runs when M * K >= 2^21 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_12a8c059b4b30e7e0f11": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_12a8c059b4b30e7e0f11",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_12a8c059b4b30e7e0f11",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_12a8c059b4b30e7e0f11",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "col",
        "select_rule": {
            "elements": {
                "sm_107a": 2097152,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with a 32 x 32 tile, 12 CTAs per SM
    # and the ffs band path (on sm_100a only): it runs when M * K >= 2^18 and
    # no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_383546f101950463dcf8": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_383546f101950463dcf8",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_383546f101950463dcf8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_383546f101950463dcf8",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 262144,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with a 32 x 32 tile, 6 CTAs per SM and
    # the hybrid band path: the route's fallback on sm_100a (no select_rule).
    "cake_gated_act_mxfp8_2ea78301fb3ef01a1b41": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_2ea78301fb3ef01a1b41",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_2ea78301fb3ef01a1b41",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_2ea78301fb3ef01a1b41",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": None,
    },
    # The backward, rowwise and colwise, with a 32 x 32 tile, 8 CTAs per SM and
    # the ffs band path (on sm_103a only): it runs when M * K >= 2^18 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_c31a8bcd7bef054bac83": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_c31a8bcd7bef054bac83",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c31a8bcd7bef054bac83",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c31a8bcd7bef054bac83",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_103a": 262144,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with a 32 x 32 tile, 8 CTAs per SM and
    # the hybrid band path: the route's fallback on sm_103a and sm_107a (no
    # select_rule).
    "cake_gated_act_mxfp8_ca9186d071397b3427fa": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 32,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_ca9186d071397b3427fa",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_ca9186d071397b3427fa",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_ca9186d071397b3427fa",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": None,
    },
    # The backward, rowwise and colwise (on sm_107a only): it runs when M * K
    # >= 2^19 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6d1956b2f454b08e6d34": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 524288,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with the ffs band path (on sm_103a
    # only): it runs when M * K >= 2^20 and no record of a higher threshold
    # admits the call.
    "cake_gated_act_mxfp8_5698f860354df3cadaed": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_5698f860354df3cadaed",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_5698f860354df3cadaed",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_5698f860354df3cadaed",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_103a": 1048576,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with 10 CTAs per SM, the packed
    # sigmoid and the ffs band path (on sm_100a and sm_103a only): it runs when
    # M * K >= 2^20 (2^22 on sm_103a) and no record of a higher threshold
    # admits the call.
    "cake_gated_act_mxfp8_905451a649b11b2d28aa": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 32,
        },
        "kernel": "cake_gated_act_mxfp8_905451a649b11b2d28aa",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_905451a649b11b2d28aa",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_905451a649b11b2d28aa",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 1048576,
                "sm_103a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, 7 CTAs per SM, the packed sigmoid and the called band path (on
    # sm_100a and sm_103a only): it runs when M * K >= 2^24 and no record of a
    # higher threshold admits the call.
    "cake_gated_act_mxfp8_44395eef5de97bba1695": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_44395eef5de97bba1695",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_44395eef5de97bba1695",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_44395eef5de97bba1695",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 16777216,
                "sm_103a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads and the hybrid band path (on sm_100a only): it runs when M * K >=
    # 2^27 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_d894c382be40fb39bc1e": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_d894c382be40fb39bc1e",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_d894c382be40fb39bc1e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_d894c382be40fb39bc1e",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_100a": 134217728,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, the packed sigmoid and the hybrid band path (on sm_103a only): it
    # runs when M * K >= 3 x 2^24 and no record of a higher threshold admits
    # the call.
    "cake_gated_act_mxfp8_69f1ccc15e17f512a07b": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_69f1ccc15e17f512a07b",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_69f1ccc15e17f512a07b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_69f1ccc15e17f512a07b",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_103a": 50331648,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, 6 CTAs per SM, the packed sigmoid and the hybrid band path (on
    # sm_107a only): it runs when M * K >= 2^23 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_f8a311b0a46a2ba0684e": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_f8a311b0a46a2ba0684e",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_f8a311b0a46a2ba0684e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_f8a311b0a46a2ba0684e",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 8388608,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, 12 CTAs per SM, the packed sigmoid and the called band path (on
    # sm_107a only): it runs when M * K >= 21 x 2^20 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_c9e9a49ccc520c8be6b0": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "gated_input"],
            ["buffer", "grad"],
            ["buffer", "row_output"],
            ["buffer", "col_output"],
            ["buffer", "row_scales"],
            ["buffer", "col_scales"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "tile_cols": 64,
            "tile_rows": 64,
        },
        "kernel": "cake_gated_act_mxfp8_c9e9a49ccc520c8be6b0",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c9e9a49ccc520c8be6b0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c9e9a49ccc520c8be6b0",
        "logical": {
            "K": "K",
            "M": "M",
            "col_output": "col_output",
            "col_scales": "col_scales",
            "gated_input": "gated_input",
            "grad": "grad_output",
            "row_output": "row_output",
            "row_scales": "row_scales",
        },
        "role": "bwd",
        "route": "both",
        "select_rule": {
            "elements": {
                "sm_107a": 22020096,
            },
            "kind": "elements_at_least",
        },
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "fwd_row": [
            "cake_gated_act_mxfp8_a097ced99ffe2f27b973",
            "cake_gated_act_mxfp8_9e34b70ff5e2423e9a42",
            "cake_gated_act_mxfp8_fd6057ad400d80a36db9",
            "cake_gated_act_mxfp8_a95dec432abb6b7d2109",
            "cake_gated_act_mxfp8_7ed7c74a790b9b536276",
            "cake_gated_act_mxfp8_baf6fbeb97918f613cef",
        ],
        "fwd_col": [
            "cake_gated_act_mxfp8_33c80e05bad314060124",
            "cake_gated_act_mxfp8_6ffdb2bd1b07f3dbf991",
            "cake_gated_act_mxfp8_6ae0a2c005d5fc854efc",
            "cake_gated_act_mxfp8_039a5afd9072a9295001",
            "cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4",
            "cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        ],
        "fwd_both": [
            "cake_gated_act_mxfp8_5e2d60e2f8ccce9e3ceb",
            "cake_gated_act_mxfp8_d1729b9c66e203234b04",
            "cake_gated_act_mxfp8_f536b838e9f5b329b548",
            "cake_gated_act_mxfp8_926bcb202902eef2465c",
        ],
        "bwd_row": [
            "cake_gated_act_mxfp8_6d93c6436dd21ee79954",
            "cake_gated_act_mxfp8_9c68fc985d41586ecbe8",
            "cake_gated_act_mxfp8_5bf054fcafaa9784cea3",
            "cake_gated_act_mxfp8_91103eb79c29acb201d7",
            "cake_gated_act_mxfp8_bdbe30b9562eca355986",
        ],
        "bwd_col": [
            "cake_gated_act_mxfp8_4a560a1480cd1898ba6c",
            "cake_gated_act_mxfp8_848a93885f596fa4cbf9",
            "cake_gated_act_mxfp8_5fe5136799b53b96cfc6",
            "cake_gated_act_mxfp8_418be6031aeee9031cfe",
            "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        ],
        "bwd_both": [
            "cake_gated_act_mxfp8_2ea78301fb3ef01a1b41",
            "cake_gated_act_mxfp8_383546f101950463dcf8",
            "cake_gated_act_mxfp8_905451a649b11b2d28aa",
            "cake_gated_act_mxfp8_44395eef5de97bba1695",
            "cake_gated_act_mxfp8_d894c382be40fb39bc1e",
        ],
    },
    "sm_103a": {
        "fwd_row": [
            "cake_gated_act_mxfp8_a097ced99ffe2f27b973",
            "cake_gated_act_mxfp8_40b0da5244b08b1db510",
            "cake_gated_act_mxfp8_e399747ca59173122fb7",
            "cake_gated_act_mxfp8_03a64d181c25359b04fd",
            "cake_gated_act_mxfp8_69fc759322d18bf4aa70",
        ],
        "fwd_col": [
            "cake_gated_act_mxfp8_33c80e05bad314060124",
            "cake_gated_act_mxfp8_7c3f4b291326f333e943",
            "cake_gated_act_mxfp8_f1232b8d8f9b3a7f66cf",
            "cake_gated_act_mxfp8_363476aea39117786ffc",
            "cake_gated_act_mxfp8_7b36d1a18f26d26d8692",
            "cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        ],
        "fwd_both": [
            "cake_gated_act_mxfp8_480a2547e3b6273df280",
            "cake_gated_act_mxfp8_03452b1f2ae3d40b1bcc",
            "cake_gated_act_mxfp8_31deb882beba0de9d64a",
            "cake_gated_act_mxfp8_ba8d72750dad06fadf5d",
        ],
        "bwd_row": [
            "cake_gated_act_mxfp8_6d93c6436dd21ee79954",
            "cake_gated_act_mxfp8_54573bbcd9b4aadbd39f",
            "cake_gated_act_mxfp8_8d0ef9eac275dd44006c",
            "cake_gated_act_mxfp8_1f02c63ba6b7d1e0a3af",
            "cake_gated_act_mxfp8_6d1f382d999ef2f444c7",
        ],
        "bwd_col": [
            "cake_gated_act_mxfp8_ede2112840c4b6e1ffff",
            "cake_gated_act_mxfp8_c2ea56613f8ca922571a",
            "cake_gated_act_mxfp8_addcbe7051b411a1e1e0",
            "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        ],
        "bwd_both": [
            "cake_gated_act_mxfp8_ca9186d071397b3427fa",
            "cake_gated_act_mxfp8_c31a8bcd7bef054bac83",
            "cake_gated_act_mxfp8_5698f860354df3cadaed",
            "cake_gated_act_mxfp8_905451a649b11b2d28aa",
            "cake_gated_act_mxfp8_44395eef5de97bba1695",
            "cake_gated_act_mxfp8_69f1ccc15e17f512a07b",
        ],
    },
    "sm_107a": {
        "fwd_row": [
            "cake_gated_act_mxfp8_a097ced99ffe2f27b973",
            "cake_gated_act_mxfp8_9e34b70ff5e2423e9a42",
            "cake_gated_act_mxfp8_7711f8af92d5b9756501",
            "cake_gated_act_mxfp8_83485c623aae181a44f7",
            "cake_gated_act_mxfp8_9e0e07c8dd55b72b2430",
            "cake_gated_act_mxfp8_81973835902ebc40b9b5",
            "cake_gated_act_mxfp8_d482e0e2dc5d60ddeb62",
        ],
        "fwd_col": [
            "cake_gated_act_mxfp8_7c3f4b291326f333e943",
            "cake_gated_act_mxfp8_0827951f83a4f724f2a0",
            "cake_gated_act_mxfp8_96497ddc9daf22dc35a5",
            "cake_gated_act_mxfp8_2e3128bb6b187c7a4a7c",
            "cake_gated_act_mxfp8_6902c8391e6f660586a4",
        ],
        "fwd_both": [
            "cake_gated_act_mxfp8_df1c4b7c00309eb06349",
            "cake_gated_act_mxfp8_d4842157b309640e4317",
            "cake_gated_act_mxfp8_0520c14fde53ef7bfea7",
            "cake_gated_act_mxfp8_c116ff68e21df5068f6d",
            "cake_gated_act_mxfp8_a133fc612a9b9a21e84b",
        ],
        "bwd_row": [
            "cake_gated_act_mxfp8_6d93c6436dd21ee79954",
            "cake_gated_act_mxfp8_8da8bc14e105ce409f74",
            "cake_gated_act_mxfp8_f8db01d14808a5afc6da",
            "cake_gated_act_mxfp8_dc95c4d586c964293ee9",
            "cake_gated_act_mxfp8_c26c9ea40f45054678ac",
        ],
        "bwd_col": [
            "cake_gated_act_mxfp8_c6ab778cd390eb44c75a",
            "cake_gated_act_mxfp8_78eaec48e0704cf3bea2",
            "cake_gated_act_mxfp8_12a8c059b4b30e7e0f11",
            "cake_gated_act_mxfp8_b1cc52ca3532a49c981e",
            "cake_gated_act_mxfp8_ca9917264bb212d23139",
            "cake_gated_act_mxfp8_49303287790d1f318420",
        ],
        "bwd_both": [
            "cake_gated_act_mxfp8_ca9186d071397b3427fa",
            "cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
            "cake_gated_act_mxfp8_f8a311b0a46a2ba0684e",
            "cake_gated_act_mxfp8_c9e9a49ccc520c8be6b0",
        ],
    },
}

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
# The routes (``ROUTES[arch]`` keys): a direction and the outputs it writes.
KINDS = ("fwd_row", "fwd_col", "fwd_both", "bwd_row", "bwd_col", "bwd_both")
# The role of a module record: a forward or a backward program.
ROLES = ("fwd", "bwd")
# The grid-rule vocabulary (``record["grid_rule"]["kind"]``): ``tiles``
# launches ``(K / tile_cols, M / tile_rows, 1)`` CTAs.
GRID_KINDS = ("tiles",)
# The selection-rule vocabulary (``record["select_rule"]["kind"]``).
SELECT_KINDS = ("elements_at_least",)
# ``M`` and ``K`` are positive multiples of this: the 128 x 4 tiles of the
# swizzled scale layout.
SHAPE_MULTIPLE = 128
# The MXFP8 block: one E8M0 scale byte per 32 elements.
BLOCK = 32
# The kernels index the input with 32-bit integers.
INT32_MAX = 2**31 - 1
# Logical argument names the host binds; every module record maps its kernel
# parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "gated_input",
    "grad_output",
    "row_output",
    "col_output",
    "row_scales",
    "col_scales",
)
LOGICAL_SCALARS = ("M", "K")


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_name(*, backward: bool, rowwise: bool, colwise: bool) -> str:
    """The route of one call: ``<fwd|bwd>_<row|col|both>`` (``ValueError`` when
    neither output is requested)."""

    if not rowwise and not colwise:
        raise ValueError("at least one of rowwise or colwise must be enabled")
    outputs = "both" if rowwise and colwise else ("row" if rowwise else "col")
    return f"{'bwd' if backward else 'fwd'}_{outputs}"


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Module names of one route on ``arch`` (every program the route may run)."""

    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake gated activation MXFP8 route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record (``None``: no threshold anywhere).

    ``{"kind": "elements_at_least", "elements": {arch: n, ...}}`` owns the
    calls with ``M * K >= elements[arch]`` on the architectures it names; when
    several records own a call, the one of the highest threshold runs.  On an
    architecture the rule names no threshold for, the record is the route's
    fallback.
    """

    return record.get("select_rule")


def _threshold(rule: dict[str, Any], *, arch: str) -> int | None:
    """The ``M * K`` from which a selection rule owns the calls on ``arch``,
    ``None`` where it never does (``ValueError`` for an unknown kind)."""

    kind = rule["kind"]
    if kind == "elements_at_least":
        elements = rule["elements"]
        return int(elements[arch]) if arch in elements else None
    raise ValueError(
        f"unknown Cake gated activation MXFP8 selection rule: {kind!r} "
        f"(known: {SELECT_KINDS})"
    )


def selected_module(arch: str, kind: str, *, M: int, K: int) -> str:
    """The module one call of route ``kind`` runs on ``arch`` for an ``[M, 2K]`` input.

    Of the records whose :func:`select_rule` admits ``M * K`` on ``arch`` the
    one of the highest threshold, else the route's single fallback record (the
    one without a threshold on ``arch``).  Two admitting records of one
    threshold, or zero or several fallbacks, are registry errors
    (``RuntimeError``).
    """

    elements = int(M) * int(K)
    positive: list[tuple[int, str]] = []
    fallback: list[str] = []
    for name in route_modules(arch, kind):
        rule = select_rule(MODULES[name])
        threshold = None if rule is None else _threshold(rule, arch=arch)
        if threshold is None:
            fallback.append(name)
        elif elements >= threshold:
            positive.append((threshold, name))
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} declares {len(fallback)} "
            "fallback programs (records without a threshold there); expected exactly one"
        )
    if not positive:
        return fallback[0]
    positive.sort(reverse=True)
    if len(positive) > 1 and positive[0][0] == positive[1][0]:
        tied = [name for threshold, name in positive if threshold == positive[0][0]]
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} selects {len(tied)} programs "
            f"for M * K = {elements} at one threshold {positive[0][0]} ({tied}); "
            "thresholds must be distinct"
        )
    return positive[0][1]


def launch_grid(record: dict[str, Any], *, M: int, K: int) -> tuple[int, int, int]:
    """The grid of one module record for an ``[M, 2K]`` input: one CTA per tile."""

    rule = record["grid_rule"]
    if rule["kind"] != "tiles":
        raise ValueError(
            f"unknown Cake gated activation MXFP8 grid rule: {rule['kind']!r} "
            f"(known: {GRID_KINDS})"
        )
    return (int(K) // int(rule["tile_cols"]), int(M) // int(rule["tile_rows"]), 1)


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            "Cake gated activation MXFP8 target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake gated activation MXFP8 module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
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


def run_module(name: str, values: dict[str, Any], *, device_index: int) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python integers; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is the
    module's :func:`launch_grid`, and the kernel runs on the current torch
    stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, M=values["M"], K=values["K"])
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
    "BLOCK",
    "GRID_KINDS",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "ROLES",
    "ROUTES",
    "SELECT_KINDS",
    "SHAPE_MULTIPLE",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "route_modules",
    "route_name",
    "run_module",
    "select_rule",
    "selected_module",
    "specs_for_arch",
]
