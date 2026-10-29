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
64 or 128 for a program whose threads own two or four rows each.  ``M`` and
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
    # The forward, rowwise, with a 32 x 32 tile: the route's fallback on
    # sm_100a and sm_103a (no select_rule).
    "cake_gated_act_mxfp8_70d1cafb2bb8a175a516": {
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
        "kernel": "cake_gated_act_mxfp8_70d1cafb2bb8a175a516",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_70d1cafb2bb8a175a516",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_70d1cafb2bb8a175a516",
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
    # The forward, rowwise, with 4 CTAs per SM and the inline band path: the
    # route's fallback on sm_107a (no select_rule).
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
        "select_rule": None,
    },
    # The forward, rowwise, with 8 CTAs per SM, the packed sigmoid and the
    # inline band path (on sm_100a only): it runs when M * K >= 2^20 and no
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
    # band path (on sm_103a only): it runs when M * K >= 2^20 and no record of
    # a higher threshold admits the call.
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
    # The forward, rowwise, with two rows per thread (on sm_107a only): it runs
    # when M * K >= 2^23 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_5b2f99c68aa8e10a26fd": {
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
        "kernel": "cake_gated_act_mxfp8_5b2f99c68aa8e10a26fd",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_5b2f99c68aa8e10a26fd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_5b2f99c68aa8e10a26fd",
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
    # The forward, rowwise, with two rows per thread, 4 CTAs per SM and the
    # called band path (on sm_107a only): it runs when M * K >= 2^24 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_75c68471d8a4921f432a": {
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
        "kernel": "cake_gated_act_mxfp8_75c68471d8a4921f432a",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_75c68471d8a4921f432a",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_75c68471d8a4921f432a",
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
                "sm_107a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise, with four rows per thread, no_allocate loads and 3
    # CTAs per SM (on sm_107a only): it runs when M * K >= 2^26 and no record
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
                "sm_107a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with a 32 x 32 tile: the route's fallback on
    # sm_100a and sm_103a (no select_rule).
    "cake_gated_act_mxfp8_7e191474abd37ba82e73": {
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
        "kernel": "cake_gated_act_mxfp8_7e191474abd37ba82e73",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_7e191474abd37ba82e73",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_7e191474abd37ba82e73",
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
    # The forward, colwise, with 4 CTAs per SM and the inline band path: the
    # route's fallback on sm_107a (no select_rule).
    "cake_gated_act_mxfp8_b1cad9a96b5541f4e8c3": {
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
        "kernel": "cake_gated_act_mxfp8_b1cad9a96b5541f4e8c3",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_b1cad9a96b5541f4e8c3",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_b1cad9a96b5541f4e8c3",
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
    # The forward, colwise, with 8 CTAs per SM and the packed sigmoid (on
    # sm_100a only): it runs when M * K >= 2^22 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_ff6e1cf61c61f4b595fa": {
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
        "kernel": "cake_gated_act_mxfp8_ff6e1cf61c61f4b595fa",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_ff6e1cf61c61f4b595fa",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_ff6e1cf61c61f4b595fa",
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
    # sm_103a only): it runs when M * K >= 2^27 (7 x 2^24 on sm_103a) and no
    # record of a higher threshold admits the call.
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
                "sm_103a": 117440512,
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
    # The forward, colwise, with two rows per thread (on sm_103a only): it runs
    # when M * K >= 2^25 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_b51156b593447fe2aadc": {
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
        "kernel": "cake_gated_act_mxfp8_b51156b593447fe2aadc",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_b51156b593447fe2aadc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_b51156b593447fe2aadc",
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
                "sm_103a": 33554432,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise, with two rows per thread, no_allocate loads, 5 CTAs
    # per SM and the ffs band path (on sm_103a only): it runs when M * K >=
    # 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_74232aea4bcc3c884369": {
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
        "kernel": "cake_gated_act_mxfp8_74232aea4bcc3c884369",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_74232aea4bcc3c884369",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_74232aea4bcc3c884369",
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
                "sm_103a": 67108864,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, colwise (on sm_107a only): it runs when M * K >= 2^20 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_f3d5bac90719f391b705": {
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
        "kernel": "cake_gated_act_mxfp8_f3d5bac90719f391b705",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_f3d5bac90719f391b705",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_f3d5bac90719f391b705",
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
    # (on sm_107a only): it runs when M * K >= 2^23 and no record of a higher
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
                "sm_107a": 8388608,
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
    # The forward, rowwise and colwise, with a 32 x 32 tile: the route's
    # fallback on sm_100a and sm_103a (no select_rule).
    "cake_gated_act_mxfp8_34e92b6e8ad60c86843d": {
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
        "kernel": "cake_gated_act_mxfp8_34e92b6e8ad60c86843d",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_34e92b6e8ad60c86843d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_34e92b6e8ad60c86843d",
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
    # The forward, rowwise and colwise: the route's fallback on sm_107a; on
    # sm_103a it runs when M * K >= 2^21 and no record of a higher threshold
    # admits the call.
    "cake_gated_act_mxfp8_0520c14fde53ef7bfea7": {
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
                "sm_103a": 2097152,
            },
            "kind": "elements_at_least",
        },
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
    # The forward, rowwise and colwise, with 8 CTAs per SM and the packed
    # sigmoid (on sm_103a only): it runs when M * K >= 2^22 and no record of a
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
                "sm_103a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The forward, rowwise and colwise, with two rows per thread, no_allocate
    # loads and 4 CTAs per SM (on sm_103a only): it runs when M * K >= 2^26 and
    # no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_c2957071a449135dded9": {
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
        "kernel": "cake_gated_act_mxfp8_c2957071a449135dded9",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_c2957071a449135dded9",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_c2957071a449135dded9",
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
    # The forward, rowwise and colwise, with two rows per thread and
    # no_allocate loads (on sm_107a only): it runs when M * K >= 2^23 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_3b4157aba492acf411db": {
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
        "kernel": "cake_gated_act_mxfp8_3b4157aba492acf411db",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_3b4157aba492acf411db",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_3b4157aba492acf411db",
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
    # loads and 3 CTAs per SM (on sm_107a only): it runs when M * K >= 3 x 2^23
    # and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_bbe04581406928a16cbc": {
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
        "kernel": "cake_gated_act_mxfp8_bbe04581406928a16cbc",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_bbe04581406928a16cbc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 32768,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_bbe04581406928a16cbc",
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
    # The backward, rowwise, with a 32 x 32 tile, 12 CTAs per SM and the called
    # band path: the route's fallback on sm_100a (no select_rule).
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
        "select_rule": None,
    },
    # The backward, rowwise, with a 32 x 32 tile, 12 CTAs per SM and the inline
    # band path: the route's fallback on sm_103a (no select_rule).
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
        "select_rule": None,
    },
    # The backward, rowwise, with a 32 x 32 tile: the route's fallback on
    # sm_107a (no select_rule).
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
        "select_rule": None,
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
    # The backward, rowwise, with no_allocate loads, 12 CTAs per SM and the ffs
    # band path (on sm_100a only): it runs when M * K >= 2^26 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_0e06658456a152d0b214": {
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
        "kernel": "cake_gated_act_mxfp8_0e06658456a152d0b214",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_0e06658456a152d0b214",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_0e06658456a152d0b214",
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
    # The backward, rowwise, with the packed sigmoid (on sm_103a only): it runs
    # when M * K >= 2^24 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_957f6d5f3e03426c0964": {
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
        "kernel": "cake_gated_act_mxfp8_957f6d5f3e03426c0964",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_957f6d5f3e03426c0964",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_957f6d5f3e03426c0964",
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
    # The backward, rowwise, with no_allocate loads, 10 CTAs per SM and the
    # packed sigmoid (on sm_103a only): it runs when M * K >= 2^26 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_3e0ac746df31b17317e0": {
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
        "kernel": "cake_gated_act_mxfp8_3e0ac746df31b17317e0",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_3e0ac746df31b17317e0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_3e0ac746df31b17317e0",
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
    # The backward, rowwise, with two rows per thread, no_allocate loads, 6
    # CTAs per SM and the ffs band path (on sm_107a only): it runs when M * K
    # >= 2^26 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_471449952da8cd17dc1b": {
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
        "kernel": "cake_gated_act_mxfp8_471449952da8cd17dc1b",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_471449952da8cd17dc1b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_471449952da8cd17dc1b",
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
    # The backward, colwise, with a 32 x 32 tile, 12 CTAs per SM and the inline
    # band path: the route's fallback on sm_100a (no select_rule).
    "cake_gated_act_mxfp8_dd19cc9df9f6cf060a97": {
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
        "kernel": "cake_gated_act_mxfp8_dd19cc9df9f6cf060a97",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_dd19cc9df9f6cf060a97",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_dd19cc9df9f6cf060a97",
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
    # The backward, colwise, with a 32 x 32 tile, 12 CTAs per SM and the ffs
    # band path: the route's fallback on sm_103a (no select_rule).
    "cake_gated_act_mxfp8_9aca7aee2d3a502d0c59": {
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
        "kernel": "cake_gated_act_mxfp8_9aca7aee2d3a502d0c59",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_9aca7aee2d3a502d0c59",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_9aca7aee2d3a502d0c59",
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
    # The backward, colwise, with the packed sigmoid: it runs when M * K >=
    # 2^21 (2^19 on sm_107a) and no record of a higher threshold admits the
    # call.
    "cake_gated_act_mxfp8_848a93885f596fa4cbf9": {
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
                "sm_103a": 2097152,
                "sm_107a": 524288,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with two rows per thread, no_allocate loads, 6
    # CTAs per SM, the packed sigmoid and the called band path: it runs when M
    # * K >= 2^24 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd": {
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
                "sm_100a": 16777216,
                "sm_103a": 16777216,
                "sm_107a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, colwise, with a 32 x 32 tile (on sm_103a only): it runs
    # when M * K >= 2^15 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_2513ecc12f03a6081c2e": {
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
        "kernel": "cake_gated_act_mxfp8_2513ecc12f03a6081c2e",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_2513ecc12f03a6081c2e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_2513ecc12f03a6081c2e",
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
    # The backward, colwise, with 8 CTAs per SM, the packed sigmoid and the ffs
    # band path (on sm_107a only): it runs when M * K >= 2^23 and no record of
    # a higher threshold admits the call.
    "cake_gated_act_mxfp8_822bead4b44c749f7afe": {
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
        "kernel": "cake_gated_act_mxfp8_822bead4b44c749f7afe",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_822bead4b44c749f7afe",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_822bead4b44c749f7afe",
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
    # CTAs per SM (on sm_107a only): it runs when M * K >= 3 x 2^25 and no
    # record of a higher threshold admits the call.
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
                "sm_107a": 100663296,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with a 32 x 32 tile, 12 CTAs per SM
    # and the ffs band path: the route's fallback on sm_100a and sm_103a (no
    # select_rule).
    "cake_gated_act_mxfp8_383546f101950463dcf8": {
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
        "select_rule": None,
    },
    # The backward, rowwise and colwise, with a 32 x 32 tile, 8 CTAs per SM and
    # the ffs band path: the route's fallback on sm_107a (no select_rule).
    "cake_gated_act_mxfp8_c31a8bcd7bef054bac83": {
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
        "select_rule": None,
    },
    # The backward, rowwise and colwise: it runs when M * K >= 2^20 (2^19 on
    # sm_107a) and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_6d1956b2f454b08e6d34": {
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
                "sm_100a": 1048576,
                "sm_103a": 1048576,
                "sm_107a": 524288,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with 10 CTAs per SM, the packed
    # sigmoid and the ffs band path (on sm_100a only): it runs when M * K >=
    # 2^22 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_905451a649b11b2d28aa": {
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
                "sm_100a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, 7 CTAs per SM, the packed sigmoid and the called band path (on
    # sm_100a only): it runs when M * K >= 2^24 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_44395eef5de97bba1695": {
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
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread and
    # no_allocate loads (on sm_100a only): it runs when M * K >= 2^27 and no
    # record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_e96f02e396eaf24e2caa": {
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
        "kernel": "cake_gated_act_mxfp8_e96f02e396eaf24e2caa",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_e96f02e396eaf24e2caa",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_e96f02e396eaf24e2caa",
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
    # The backward, rowwise and colwise, with 10 CTAs per SM and the packed
    # sigmoid (on sm_103a only): it runs when M * K >= 2^22 and no record of a
    # higher threshold admits the call.
    "cake_gated_act_mxfp8_f4e33162da7e5d60c9b4": {
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
        "kernel": "cake_gated_act_mxfp8_f4e33162da7e5d60c9b4",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_f4e33162da7e5d60c9b4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8192,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_f4e33162da7e5d60c9b4",
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
                "sm_103a": 4194304,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads and the packed sigmoid (on sm_103a only): it runs when M * K >=
    # 2^24 and no record of a higher threshold admits the call.
    "cake_gated_act_mxfp8_eb924283e8b660bdeea8": {
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
        "kernel": "cake_gated_act_mxfp8_eb924283e8b660bdeea8",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_eb924283e8b660bdeea8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_eb924283e8b660bdeea8",
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
                "sm_103a": 16777216,
            },
            "kind": "elements_at_least",
        },
    },
    # The backward, rowwise and colwise, with two rows per thread, no_allocate
    # loads, 6 CTAs per SM, the packed sigmoid and the ffs band path (on
    # sm_107a only): it runs when M * K >= 2^23 and no record of a higher
    # threshold admits the call.
    "cake_gated_act_mxfp8_43b7d4d5cc26486e8036": {
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
        "kernel": "cake_gated_act_mxfp8_43b7d4d5cc26486e8036",
        "kernel_symbol": "kernel_cake_gated_act_mxfp8_43b7d4d5cc26486e8036",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": False,
        },
        "library": "cake_gated_act_mxfp8_43b7d4d5cc26486e8036",
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
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "fwd_row": [
            "cake_gated_act_mxfp8_70d1cafb2bb8a175a516",
            "cake_gated_act_mxfp8_7ed7c74a790b9b536276",
            "cake_gated_act_mxfp8_baf6fbeb97918f613cef",
        ],
        "fwd_col": [
            "cake_gated_act_mxfp8_7e191474abd37ba82e73",
            "cake_gated_act_mxfp8_ff6e1cf61c61f4b595fa",
            "cake_gated_act_mxfp8_039a5afd9072a9295001",
            "cake_gated_act_mxfp8_0b9dfb9aa4b837a6b7a4",
            "cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        ],
        "fwd_both": [
            "cake_gated_act_mxfp8_34e92b6e8ad60c86843d",
            "cake_gated_act_mxfp8_f536b838e9f5b329b548",
            "cake_gated_act_mxfp8_926bcb202902eef2465c",
        ],
        "bwd_row": [
            "cake_gated_act_mxfp8_9c68fc985d41586ecbe8",
            "cake_gated_act_mxfp8_5bf054fcafaa9784cea3",
            "cake_gated_act_mxfp8_0e06658456a152d0b214",
            "cake_gated_act_mxfp8_bdbe30b9562eca355986",
        ],
        "bwd_col": [
            "cake_gated_act_mxfp8_dd19cc9df9f6cf060a97",
            "cake_gated_act_mxfp8_848a93885f596fa4cbf9",
            "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        ],
        "bwd_both": [
            "cake_gated_act_mxfp8_383546f101950463dcf8",
            "cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
            "cake_gated_act_mxfp8_905451a649b11b2d28aa",
            "cake_gated_act_mxfp8_44395eef5de97bba1695",
            "cake_gated_act_mxfp8_e96f02e396eaf24e2caa",
        ],
    },
    "sm_103a": {
        "fwd_row": [
            "cake_gated_act_mxfp8_70d1cafb2bb8a175a516",
            "cake_gated_act_mxfp8_03a64d181c25359b04fd",
            "cake_gated_act_mxfp8_69fc759322d18bf4aa70",
        ],
        "fwd_col": [
            "cake_gated_act_mxfp8_7e191474abd37ba82e73",
            "cake_gated_act_mxfp8_f1232b8d8f9b3a7f66cf",
            "cake_gated_act_mxfp8_363476aea39117786ffc",
            "cake_gated_act_mxfp8_7b36d1a18f26d26d8692",
            "cake_gated_act_mxfp8_b51156b593447fe2aadc",
            "cake_gated_act_mxfp8_74232aea4bcc3c884369",
            "cake_gated_act_mxfp8_b7975e9eda41aefb0456",
        ],
        "fwd_both": [
            "cake_gated_act_mxfp8_34e92b6e8ad60c86843d",
            "cake_gated_act_mxfp8_0520c14fde53ef7bfea7",
            "cake_gated_act_mxfp8_03452b1f2ae3d40b1bcc",
            "cake_gated_act_mxfp8_c2957071a449135dded9",
            "cake_gated_act_mxfp8_ba8d72750dad06fadf5d",
        ],
        "bwd_row": [
            "cake_gated_act_mxfp8_54573bbcd9b4aadbd39f",
            "cake_gated_act_mxfp8_8d0ef9eac275dd44006c",
            "cake_gated_act_mxfp8_957f6d5f3e03426c0964",
            "cake_gated_act_mxfp8_3e0ac746df31b17317e0",
        ],
        "bwd_col": [
            "cake_gated_act_mxfp8_9aca7aee2d3a502d0c59",
            "cake_gated_act_mxfp8_2513ecc12f03a6081c2e",
            "cake_gated_act_mxfp8_848a93885f596fa4cbf9",
            "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
        ],
        "bwd_both": [
            "cake_gated_act_mxfp8_383546f101950463dcf8",
            "cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
            "cake_gated_act_mxfp8_f4e33162da7e5d60c9b4",
            "cake_gated_act_mxfp8_eb924283e8b660bdeea8",
        ],
    },
    "sm_107a": {
        "fwd_row": [
            "cake_gated_act_mxfp8_7711f8af92d5b9756501",
            "cake_gated_act_mxfp8_5b2f99c68aa8e10a26fd",
            "cake_gated_act_mxfp8_75c68471d8a4921f432a",
            "cake_gated_act_mxfp8_d482e0e2dc5d60ddeb62",
        ],
        "fwd_col": [
            "cake_gated_act_mxfp8_b1cad9a96b5541f4e8c3",
            "cake_gated_act_mxfp8_f3d5bac90719f391b705",
            "cake_gated_act_mxfp8_96497ddc9daf22dc35a5",
            "cake_gated_act_mxfp8_6902c8391e6f660586a4",
        ],
        "fwd_both": [
            "cake_gated_act_mxfp8_0520c14fde53ef7bfea7",
            "cake_gated_act_mxfp8_3b4157aba492acf411db",
            "cake_gated_act_mxfp8_bbe04581406928a16cbc",
        ],
        "bwd_row": [
            "cake_gated_act_mxfp8_8da8bc14e105ce409f74",
            "cake_gated_act_mxfp8_f8db01d14808a5afc6da",
            "cake_gated_act_mxfp8_471449952da8cd17dc1b",
        ],
        "bwd_col": [
            "cake_gated_act_mxfp8_c6ab778cd390eb44c75a",
            "cake_gated_act_mxfp8_848a93885f596fa4cbf9",
            "cake_gated_act_mxfp8_822bead4b44c749f7afe",
            "cake_gated_act_mxfp8_6ee084f4ded4d19cd8cd",
            "cake_gated_act_mxfp8_49303287790d1f318420",
        ],
        "bwd_both": [
            "cake_gated_act_mxfp8_c31a8bcd7bef054bac83",
            "cake_gated_act_mxfp8_6d1956b2f454b08e6d34",
            "cake_gated_act_mxfp8_43b7d4d5cc26486e8036",
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
