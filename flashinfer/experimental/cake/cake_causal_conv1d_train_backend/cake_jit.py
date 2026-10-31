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

Host plan and loader glue of the Cake packed width-4 depthwise causal
convolution + SiLU training kernels of the non-deterministic package (SM100 /
SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch
record, grid rule, workspace rule and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it,
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

The routes, their selection rules and the forward programs are those of the
deterministic package (``flashinfer.cake_causal_conv1d_det_train_backend``):
the forward routes ``fwd`` / ``fwd_dt`` / ``fwd_bias`` run this package's
copies of the det forward programs (no cross-CTA reduction, so ``y`` is the
det program's bits), and the backward routes ``bwd`` / ``bwd_dt`` /
``bwd_bias`` / ``bwd_fp32g`` / ``bwd_bias_fp32g`` select, by the same channel
(and token) thresholds, single-launch builds of the det backward bodies.  A
backward call is ONE ordinary launch: every per-thread and per-CTA FP32 step
and ``dx`` are the det program's, and only the cross-CTA reduction of
``dweight`` / ``dbias`` differs.  Instead of storing one partial row per CTA
for a reduce program to sum in a fixed order, each CTA adds its CTA row into
the accumulators (logical tensor ``acc``, kernel parameter ``dw_acc``) with
unordered global reductions -- FP32 ``red.add``
(``v4``) where the det reduce sums in FP32 (``bwd``, ``bwd_dt``,
``bwd_bias``), FP64 ``atomicAdd`` where it sums in FP64 (the ``_fp32g``
routes) -- then takes a ticket on its channel tile's ``work`` counter; the
tile's last CTA reads the tile's accumulators, stores ``dw`` (and ``db``) in
the dtypes ``flags`` names (one rounding from the accumulator), and stores
zeros back.  So ``acc`` and ``work`` are zero before and after every launch
(CUDA Graph replays included), the host never zeroes them again, and nothing
synchronizes with the host.  The summation order of the cross-CTA sum depends
on the CTAs' timing: ``dweight`` / ``dbias`` are not bitwise reproducible run
to run (the only relaxed property); their accuracy budget is the det
reduction's.

The backward workspace (:func:`accumulator_layout`) is one ``uint8`` buffer:
``acc`` (``taps x C`` FP32 or FP64 words: ``[C, 4]`` taps, then ``[C]`` bias
on the bias routes) at offset 0 and the uint32 ``work`` counters (one per
channel tile) at the next 16-byte boundary.  Its size depends on ``C`` and the
route, never on ``S`` or the grid; the public size
(:func:`backward_workspace_bytes`) is the largest over every backward program
of the architecture, so one zeroed buffer serves every route.

The program name of a forward copy is ``cake_causal_conv1d_train_<20 hex>``
(the det program's identity digits under this package's prefix); the backward
programs' names are the identity digests of their own kernel module key (so no
backward name equals a name the det package ships).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_f5497330bfd286211643`` (built with ``fast=True, sched="dynamic",
    # chunk_stages=1, pair_math=True``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's); the route's fallback (no ``select_rule``).  ``work`` holds the int32 token-chunk counters (two
    # per channel tile), zero before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_f5497330bfd286211643": {
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
            ["buffer", "work"],
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
        "kernel": "cake_causal_conv1d_train_f5497330bfd286211643",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f5497330bfd286211643",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_f5497330bfd286211643",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_9bfa65707a49df6a48ce`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host
    # synchronization, no cooperative launch.
    "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
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
        "kernel": "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_b4f7ff5a3dd9ddd2f41c`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host
    # synchronization, no cooperative launch.
    "cake_causal_conv1d_train_a79ed3e33712abf525ad": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
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
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_9a07bcaa0223632640ee`` (the same build keywords, body and grid; selected
    # from C = 256 on sm_107a and S >= 262144 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx``
    # are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the
    # FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce
    # program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_5ed71ee5908c07376183": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
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
        "kernel": "cake_causal_conv1d_train_5ed71ee5908c07376183",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_5ed71ee5908c07376183",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_5ed71ee5908c07376183",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 256,
            },
            "kind": "channels_tokens_at_least",
            "tokens": {
                "sm_107a": 262144,
            },
        },
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_37e428406a0cc4a8ca1b`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 768 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and
    # ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into
    # the FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce
    # program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_461d358d7090642c65de": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
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
        "kernel": "cake_causal_conv1d_train_461d358d7090642c65de",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_461d358d7090642c65de",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_461d358d7090642c65de",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 768,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_d46036aa2bc5f90bb0f6`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_103a, 512 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise
    # the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32
    # accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce
    # program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
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
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_103a": 512,
                "sm_107a": 512,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_aeff6106e2822f313a0c`` (the same build keywords, body and grid; selected
    # from C = 2048 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host
    # synchronization, no cooperative launch.
    "cake_causal_conv1d_train_c798878a383c652c428d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
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
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_c798878a383c652c428d",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_c798878a383c652c428d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_c798878a383c652c428d",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 2048,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_9c1e7ee9f37e3102660e`` (built with ``dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=1, pair_math=True``; the same kernel under this package's prefix, so ``y``
    # is bitwise the det program's); the route's fallback (no ``select_rule``).  ``work`` holds the int32 token-
    # chunk counters (two per channel tile), zero before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_9c1e7ee9f37e3102660e": {
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
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_08880ca12e4064bc8cdb`` (built with ``bias=True, dtype_flags=True,
    # fast=True, sched="dynamic", chunk_stages=1, pair_math=True``; the same kernel under this package's prefix,
    # so ``y`` is bitwise the det program's); the route's fallback (no ``select_rule``).  ``work`` holds the
    # int32 token-chunk counters (two per channel tile), zero before the first launch and left zero by every
    # launch.
    "cake_causal_conv1d_train_08880ca12e4064bc8cdb": {
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
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_8665d0756b8240ecb4d0`` (built with ``fast=True, sched="dynamic",
    # chunk_stages=2, pair_math=True``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's); selected from C = 256 on sm_107a.  ``work`` holds the int32 token-chunk counters (two per
    # channel tile), zero before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_8665d0756b8240ecb4d0": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
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
        "kernel": "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 256,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_176ebb3e14f7ad9c94dc`` (built with ``dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=2, pair_math=True``; the same kernel under this package's prefix, so ``y``
    # is bitwise the det program's); selected from C = 512 on sm_107a.  ``work`` holds the int32 token-chunk
    # counters (two per channel tile), zero before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 512,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_d829c95e1d610bc14b23`` (built with ``bias=True, dtype_flags=True,
    # fast=True, sched="dynamic", chunk_stages=2, pair_math=True``; the same kernel under this package's prefix,
    # so ``y`` is bitwise the det program's); selected from C = 1024 on sm_107a.  ``work`` holds the int32
    # token-chunk counters (two per channel tile), zero before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_d829c95e1d610bc14b23": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``bwd_dt``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_9c8310a1d3fa9b8ccffa`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from
    # FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after every
    # launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_32c5625f33586925f154": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_32c5625f33586925f154",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_32c5625f33586925f154",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_32c5625f33586925f154",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_dt``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_8b9b8db0833fd64a578f`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from
    # FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after every
    # launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_d7f458b69c0249f63c37": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_d7f458b69c0249f63c37",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d7f458b69c0249f63c37",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_d7f458b69c0249f63c37",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_2d91e5f926238453c78c`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work`` counter
    # (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags``
    # names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are
    # zero before and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_befc06774f6fee7c36ce": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_befc06774f6fee7c36ce",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_befc06774f6fee7c36ce",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_befc06774f6fee7c36ce",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_cc7a117aee9c7ae829c4`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work`` counter
    # (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags``
    # names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are
    # zero before and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_17d30c46fe34f0266e26": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_17d30c46fe34f0266e26",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_17d30c46fe34f0266e26",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_17d30c46fe34f0266e26",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_c8027a64ba3cc38333bf`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP64 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from
    # FP64) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after every
    # launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_c3d9c23921f5573d0577": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_c3d9c23921f5573d0577",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_c3d9c23921f5573d0577",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_c3d9c23921f5573d0577",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float64",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_f1545adfd2bb28044419`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP64 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from
    # FP64) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after every
    # launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_064063666308a8552401": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_064063666308a8552401",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_064063666308a8552401",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_064063666308a8552401",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float64",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_149ca6f53934ba630033`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP64 accumulator
    # ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work`` counter
    # (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags``
    # names once (one rounding from FP64) and zeroes its accumulators and counter, so ``acc`` / ``work`` are
    # zero before and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_a47e12df7005fb5fb72b": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_a47e12df7005fb5fb72b",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a47e12df7005fb5fb72b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_a47e12df7005fb5fb72b",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float64",
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_843c010db78c79107fd3`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP64 accumulator
    # ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work`` counter
    # (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags``
    # names once (one rounding from FP64) and zeroes its accumulators and counter, so ``acc`` / ``work`` are
    # zero before and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_61b7fb3c4a7d481011eb": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_61b7fb3c4a7d481011eb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_61b7fb3c4a7d481011eb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_61b7fb3c4a7d481011eb",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float64",
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_dt``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_2675cc1cb4efc0937378`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and
    # ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into
    # the FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once
    # (one rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before
    # and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_012b6319a0cc717911b9": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_012b6319a0cc717911b9",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_012b6319a0cc717911b9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_012b6319a0cc717911b9",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
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
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_517710d1f0988cf570a9`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and
    # ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into
    # the FP32 accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's
    # ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the
    # dtypes ``flags`` names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc`` /
    # ``work`` are zero before and after every launch.  No reduce program, no host synchronization, no
    # cooperative launch.
    "cake_causal_conv1d_train_383cb5839559d5b0e72b": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_383cb5839559d5b0e72b",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_383cb5839559d5b0e72b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 87040,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_383cb5839559d5b0e72b",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
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
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_170d89481a616ae977b3`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and
    # ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into
    # the FP64 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once
    # (one rounding from FP64) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before
    # and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_d542ad2292353c56bd65": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_d542ad2292353c56bd65",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d542ad2292353c56bd65",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_d542ad2292353c56bd65",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float64",
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_a681165b3e81204c73ff`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and
    # ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into
    # the FP64 accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's
    # ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the
    # dtypes ``flags`` names once (one rounding from FP64) and zeroes its accumulators and counter, so ``acc`` /
    # ``work`` are zero before and after every launch.  No reduce program, no host synchronization, no
    # cooperative launch.
    "cake_causal_conv1d_train_0cc1521134686f49dbc9": {
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
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
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
        "kernel": "cake_causal_conv1d_train_0cc1521134686f49dbc9",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_0cc1521134686f49dbc9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_0cc1521134686f49dbc9",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float64",
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
}

# architecture -> route -> the records the route may launch (fallback first).
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd": [
            "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
            "cake_causal_conv1d_train_461d358d7090642c65de",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_befc06774f6fee7c36ce",
            "cake_causal_conv1d_train_383cb5839559d5b0e72b",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_a47e12df7005fb5fb72b",
            "cake_causal_conv1d_train_0cc1521134686f49dbc9",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_32c5625f33586925f154",
            "cake_causal_conv1d_train_012b6319a0cc717911b9",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_c3d9c23921f5573d0577",
            "cake_causal_conv1d_train_d542ad2292353c56bd65",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f5497330bfd286211643",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        ],
    },
    "sm_103a": {
        "bwd": [
            "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_befc06774f6fee7c36ce",
            "cake_causal_conv1d_train_383cb5839559d5b0e72b",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_a47e12df7005fb5fb72b",
            "cake_causal_conv1d_train_0cc1521134686f49dbc9",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_32c5625f33586925f154",
            "cake_causal_conv1d_train_012b6319a0cc717911b9",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_c3d9c23921f5573d0577",
            "cake_causal_conv1d_train_d542ad2292353c56bd65",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f5497330bfd286211643",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        ],
    },
    "sm_107a": {
        "bwd": [
            "cake_causal_conv1d_train_a79ed3e33712abf525ad",
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
            "cake_causal_conv1d_train_c798878a383c652c428d",
            "cake_causal_conv1d_train_5ed71ee5908c07376183",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_17d30c46fe34f0266e26",
            "cake_causal_conv1d_train_383cb5839559d5b0e72b",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_61b7fb3c4a7d481011eb",
            "cake_causal_conv1d_train_0cc1521134686f49dbc9",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_d7f458b69c0249f63c37",
            "cake_causal_conv1d_train_012b6319a0cc717911b9",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_064063666308a8552401",
            "cake_causal_conv1d_train_d542ad2292353c56bd65",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f5497330bfd286211643",
            "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
            "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
            "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
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
# The routes (``ROUTES[arch]`` keys) a public call may resolve to.  ``fwd`` /
# ``bwd`` are the default routes (FP32 taps, no bias, SiLU' at the BF16-rounded
# preactivation); ``_dt`` routes take BF16 taps and / or store a BF16 gradient
# (the ``flags`` scalar says which), ``_bias`` routes add the per-channel bias
# and its gradient, ``_fp32g`` routes evaluate SiLU' at the unrounded FP32
# preactivation for the weight gradient.  :func:`forward_route` and
# :func:`backward_route` are the host's map from a call's arguments to a route.
KINDS = (
    "fwd",
    "fwd_dt",
    "fwd_bias",
    "bwd",
    "bwd_dt",
    "bwd_bias",
    "bwd_fp32g",
    "bwd_bias_fp32g",
)
# The routes every exported architecture ships: the base forward and backward.
# The other routes are optional per architecture; a call that needs one the
# architecture lacks is refused with a ``ValueError`` naming the route.
REQUIRED_KINDS = ("fwd", "bwd")
# The role of a module record: a forward program or a backward program (the
# one launch of a backward call; this package has no reduce programs).
ROLES = ("fwd", "bwd")
# The grid-rule vocabulary (``record["grid_rule"]["kind"]``, see
# :func:`launch_grid`): ``co_resident`` launches exactly ``ctas_per_sm x SM
# count`` CTAs, the co-resident count (the forward records and the KCP forward
# and head programs of :mod:`.kcp_jit`, which share this host plan);
# ``occupancy`` launches ``grid_mult[arch] x ctas_per_sm x SM count`` CTAs
# normally (the backward programs, single-rank and KCP; ``grid_mult`` names
# every architecture of its record).
GRID_KINDS = ("co_resident", "occupancy")
# The selection-rule vocabulary of a forward or backward record
# (``record["select_rule"]["kind"]``); of the records whose rule admits a call
# the one of the highest channel threshold runs, and a record without a rule is
# its route's fallback.  ``channels_tokens_at_least`` adds a token floor to the
# channel threshold.
SELECT_KINDS = ("channels_at_least", "channels_tokens_at_least")
# The ``flags`` bit mask of the ``dtype_flags`` programs (every program of the
# six round-6 routes): the host derives it from the tap tensor, the ACTUAL
# ``dweight`` out tensor, the bias tensor and the ACTUAL ``dbias`` out tensor;
# the kernel reads it once in its prologue (load form) and once in its epilogue
# (store form).  The default routes' programs take no ``flags``.
FLAG_W_BF16 = 1  # ``w`` holds BF16 taps (else FP32)
FLAG_DW_BF16 = 2  # ``dw`` is stored as BF16 (else FP32)
FLAG_B_BF16 = 4  # ``b`` holds BF16 (else FP32); bias programs
FLAG_DB_BF16 = 8  # ``db`` is stored as BF16 (else FP32); bias programs
# The convolution width: ``weight`` and ``dweight`` are ``[C, WIDTH]`` and tap
# ``WIDTH - 1`` multiplies the current token.
WIDTH = 4
# A forward thread and a backward thread each own one 16-byte vector of
# eight BF16 channels, so ``C`` is a multiple of eight; the ``vec`` of every
# record's geometry divides this.
CHANNEL_MULTIPLE = 8
# The kernels take ``S``, ``C`` and the plan scalars as 32-bit integers.
INT32_MAX = 2**31 - 1
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "cu",
    "y",
    "dy",
    "dx",
    "dw",
    "acc",
    "b",
    "db",
    "work",
)
# The workspace-rule vocabulary (``record["workspace_rule"]["kind"]``): a
# backward program's ``accumulators`` (``{"kind": "accumulators", "taps": 4 | 5,
# "acc": "float32" | "float64", "work": "uint32"[, "slots": K]}``:
# ``slots x taps x C`` accumulator words of dtype ``acc`` -- ``slots`` copies
# (the program's accumulator copies; 1 when absent) of the ``[C, 4]`` taps, then
# the ``[C]`` bias on the bias routes -- and one ``work`` ticket counter per
# channel tile,
# :func:`accumulator_layout`) and a forward program's ``tile_counters`` (its
# int32 ``work`` counters, ``words_per_tile`` per channel tile,
# :func:`counter_words`).  The backward's ``work`` and the forward's ``work`` are
# separate buffers of separate calls.
WORKSPACE_KINDS = ("accumulators", "tile_counters")
# The dtype names an ``accumulators`` rule may give ``acc`` and ``work``, with
# their element sizes in bytes.
ACCUMULATOR_DTYPES: dict[str, int] = {"float32": 4, "float64": 8}
COUNTER_DTYPES: dict[str, int] = {"uint32": 4}
# The byte alignment of the ``acc`` and ``work`` regions inside the backward
# workspace (the kernels add 16-byte ``v4`` vectors into ``acc``).
WORKSPACE_ALIGN = 16
# The scalars :func:`plan` produces, in kernel order; ``flags`` (the dtype bit
# mask of the ``dtype_flags`` programs) is set by the host beside them.
PLAN_SCALARS = (
    "S",
    "C",
    "num_docs",
    "search_steps",
    "ct_vec",
    "n_sub",
    "n_ch_tiles",
)
LOGICAL_SCALARS = (*PLAN_SCALARS, "flags")


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_applies(*, device_capability: Sequence[int], kind: str) -> bool:
    """Whether the verified export owns one training call on a device."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return kind in ROUTES.get(arch, {})


def route_role(kind: str) -> str:
    """``"fwd"`` or ``"bwd"``: the kernel a route's programs are built from (``ValueError`` for an unknown route)."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return "fwd" if kind.startswith("fwd") else "bwd"


def forward_route(*, bias: bool, bf16_taps: bool) -> str:
    """The forward route of a call: ``fwd_bias`` with a bias, else ``fwd_dt`` for BF16 taps, else ``fwd``.

    A bias program always has a bias (no runtime switch: adding ``+0.0`` would
    flip the sign of ``-0.0`` outputs), so ``bias=None`` never selects one; the
    bias programs read FP32 or BF16 taps and bias through ``flags``.
    """

    if bias:
        return "fwd_bias"
    return "fwd_dt" if bf16_taps else "fwd"


def backward_route(*, bias: bool, fp32_grad: bool, bf16_taps_or_dweight: bool) -> str:
    """The backward route of a call.

    ``bias`` selects ``bwd_bias`` / ``bwd_bias_fp32g`` (by ``fp32_grad``, the
    ``silu_grad_preactivation="fp32"`` policy); without a bias ``fp32_grad``
    selects ``bwd_fp32g``; else BF16 taps or a BF16 ``dweight`` out tensor
    select ``bwd_dt``; else the default ``bwd``.  Every route but ``bwd``
    reads the tap / gradient dtypes from ``flags``.
    """

    if bias:
        return "bwd_bias_fp32g" if fp32_grad else "bwd_bias"
    if fp32_grad:
        return "bwd_fp32g"
    return "bwd_dt" if bf16_taps_or_dweight else "bwd"


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch).

    A call launches the one :func:`selected_modules` picks for its channel
    count.
    """

    route_role(kind)
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record, ``None`` when the record carries none.

    A record without a rule (no ``select_rule`` key, or ``None``) is its
    route's fallback program.  ``{"kind": "channels_at_least", "channels":
    {arch: C_min, ...}}`` owns the calls whose channel count is at least
    ``channels[arch]`` on the architectures it names and never hits on the
    others.  ``{"kind": "channels_tokens_at_least", "channels": {arch: C_min,
    ...}, "tokens": {arch: S_min, ...}}`` owns those calls only when the token
    count is also at least ``tokens[arch]``.  When several records own a call,
    the one of the highest channel threshold runs.
    """

    return record.get("select_rule")


def _threshold(rule: dict[str, Any], *, arch: str) -> int | None:
    """The channel count from which a selection rule owns the calls on ``arch``, ``None`` where it never does
    (``ValueError`` for an unknown kind)."""

    kind = rule["kind"]
    if kind in SELECT_KINDS:
        channels = rule["channels"]
        return int(channels[arch]) if arch in channels else None
    raise ValueError(
        f"unknown Cake causal conv1d training selection rule: {kind!r} "
        f"(known: {SELECT_KINDS})"
    )


def _tokens(rule: dict[str, Any], *, arch: str) -> int | None:
    """The token count from which a selection rule owns the calls on ``arch`` (its token floor), ``None`` when the
    rule has none there (``ValueError`` for an unknown kind)."""

    _threshold(rule, arch=arch)
    if rule["kind"] != "channels_tokens_at_least":
        return None
    tokens = rule["tokens"]
    return int(tokens[arch]) if arch in tokens else None


def _admits(rule: dict[str, Any], *, arch: str, C: int, S: int | None = None) -> bool:
    """Whether a selection rule owns ``C`` channels (and ``S`` tokens) on ``arch``.

    ``ValueError`` for an unknown kind, or when the rule's token floor decides
    and ``S`` is ``None``.
    """

    threshold = _threshold(rule, arch=arch)
    if threshold is None or int(C) < threshold:
        return False
    floor = _tokens(rule, arch=arch)
    if floor is None:
        return True
    if S is None:
        raise ValueError(
            f"the Cake causal conv1d training selection rule {rule!r} on {arch} "
            f"for C={int(C)} runs from S {floor}; the selection needs S"
        )
    return int(S) >= floor


def selected_modules(
    arch: str, kind: str, *, C: int, S: int | None = None
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``C`` channels and ``S`` tokens.

    A route may carry several programs of its kernel and launches exactly one:
    of the records whose :func:`select_rule` admits the call on ``arch`` the
    one of the highest channel threshold, else the route's single fallback
    record (the one without a rule).  The choice is a pure function of the
    architecture, the route, ``C`` and ``S`` -- ``S`` enters only through a
    rule's token floor, and the host never reads the data -- so a given ``(S,
    C, device)`` always runs the same program of a route (``dx`` is that
    program's bits; ``dweight`` / ``dbias`` are summed in an unordered
    cross-CTA order).  Two records of one channel threshold that
    both reach ``C``, or zero or several fallbacks, are registry errors
    (``RuntimeError``); an unknown rule kind, or ``S`` of ``None`` where a
    token floor decides, is a ``ValueError``.
    """

    names = route_modules(arch, kind)
    candidates: list[tuple[int, str]] = []
    fallback: list[str] = []
    for name in names:
        rule = select_rule(MODULES[name])
        if rule is None:
            fallback.append(name)
            continue
        threshold = _threshold(rule, arch=arch)
        if threshold is not None and int(C) >= threshold:
            candidates.append((threshold, name))
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} declares {len(fallback)} "
            "fallback programs (records without a select_rule); expected exactly one"
        )
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        tied = [name for threshold, name in candidates if threshold == candidates[0][0]]
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} selects {len(tied)} "
            f"programs for C={int(C)} at one threshold {candidates[0][0]} ({tied}); "
            "thresholds must be distinct"
        )
    for _threshold_c, name in candidates:
        if _admits(select_rule(MODULES[name]), arch=arch, C=C, S=S):
            return (name,)
    return (fallback[0],)


def backward_kinds(arch: str) -> tuple[str, ...]:
    """The backward routes ``arch`` exports, in :data:`KINDS` order."""

    return tuple(
        kind
        for kind in KINDS
        if route_role(kind) == "bwd" and ROUTES.get(arch, {}).get(kind)
    )


def workspace_owners(arch: str) -> tuple[dict[str, Any], ...]:
    """The backward module records of ``arch`` (each owns an ``accumulators`` workspace), over every backward route
    in :data:`KINDS` order, each route in launch order.

    Every backward program declares an ``accumulators`` rule; an architecture
    without a backward program, or a backward record of another rule, is a
    registry error (``RuntimeError``).
    """

    owners = tuple(
        MODULES[name]
        for kind in backward_kinds(arch)
        for name in route_modules(arch, kind)
    )
    if not owners:
        raise RuntimeError(
            f"the exported backward routes on {arch} declare no workspace owner; "
            "expected at least one"
        )
    for owner in owners:
        rule = owner["workspace_rule"]
        if rule is None or rule["kind"] != "accumulators":
            raise RuntimeError(
                f"the backward program {owner['kernel']} on {arch} declares the workspace rule "
                f"{rule!r}; every backward program of this package owns an 'accumulators' workspace"
            )
    return owners


def workspace_record(arch: str) -> dict[str, Any]:
    """The capacity owner of the backward workspace on ``arch``.

    The public workspace serves every backward program of every backward route
    (:func:`workspace_owners`).  The owners share one build geometry, so for
    every ``C`` they plan one ``n_ch_tiles`` and their workspaces differ only
    in the accumulator bytes per channel, ``taps x sizeof(acc)``: the record
    with the most (an FP64 bias route's 40 bytes per channel; :data:`KINDS`
    then route order break ties) bounds them all.  Owners of different
    geometries are a registry error (``RuntimeError``).
    """

    owners = workspace_owners(arch)
    geometries = {
        tuple(sorted(owner["grid_rule"]["geometry"].items())) for owner in owners
    }
    if len(geometries) != 1:
        raise RuntimeError(
            f"the backward programs on {arch} declare {len(geometries)} build "
            "geometries; one workspace serves them only when they share one"
        )
    best = owners[0]
    for owner in owners[1:]:
        if _accumulator_bytes_per_channel(owner) > _accumulator_bytes_per_channel(best):
            best = owner
    return best


def grid_mult(record: dict[str, Any], arch: str) -> int:
    """The over-decomposition factor of an ``occupancy`` record on ``arch`` (1 under every other grid rule).

    ``grid_rule["grid_mult"]`` maps each architecture of the record to the
    factor its grid multiplies the co-resident CTA count by (1 for every
    program of this package: the det package's measured factor, kept).
    """

    rule = record["grid_rule"]
    if rule["kind"] != "occupancy":
        return 1
    factor = int(rule["grid_mult"][arch])
    if factor < 1:
        raise ValueError(
            f"module {record['kernel']}: grid_mult[{arch!r}] must be positive, got {factor}"
        )
    return factor


def grid_per_sm(record: dict[str, Any], arch: str) -> int:
    """CTAs per SM of a ``co_resident`` or ``occupancy`` record's grid on ``arch``:
    ``grid_mult x ctas_per_sm`` (the grid is this times the SM count)."""

    return grid_mult(record, arch) * int(record["grid_rule"]["ctas_per_sm"])


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


# The roles whose programs plan their channel tiles with :func:`channel_split`
# (the KCP programs plan with ``legacy=True``): each has its own rule.
SPLIT_ROLES = ("fwd", "bwd")


def channel_split(
    C: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of ``C`` channels under a record's build geometry and role.

    ``ct_vec`` channel vectors per channel tile (one thread each), ``n_sub =
    threads // ct_vec`` token sub-tiles per CTA and ``n_ch_tiles = ceil(C / vec
    / ct_vec)`` channel tiles.  ``role`` is the record's, ``"fwd"`` or
    ``"bwd"`` (a ``ValueError`` without one unless ``legacy``): the two rules
    agree except in the fallback below.  With ``cv = C / vec`` channel
    vectors: ``ct_vec = cv`` when ``cv <= threads``; above that, the multiple
    of 32 dividing ``cv`` (at most ``threads``) with the highest lane use
    ``ct_vec x (threads // ct_vec)``, the larger ``ct_vec`` on a tie.  So C =
    1280 (160 vectors) runs five 32-vector tiles with four token sub-tiles
    each, (32, 4, 5), and C = 1536 (64, 2, 3) -- every lane busy -- instead of
    two 128-vector tiles at 62 % / 75 % lane use; C <= 1024 and 2048 / 4096 /
    8192 are unchanged.  When no multiple of 32 divides ``cv``, the role's
    fallback picks the ``ct_vec`` of the highest lane use over the launch,
    ``cv / (n_ch_tiles x ct_vec) x (ct_vec x (threads // ct_vec) / threads)``
    rounded to 9 decimals (the larger on a tie), among every multiple of 32 at
    most ``threads`` (the last tile partial) and, for the forward, every
    divisor of ``cv`` from 40 to ``threads`` (whole tiles).  At 128 x 8 the
    backward runs C = 1160 as (32, 4, 5), 1344 as (64, 2, 3) and 1600 as (32,
    4, 7), the forward 1160 as (32, 4, 5), 1344 as (42, 3, 4) and 1600 as (40,
    3, 5); both run 1920 as (128, 1, 2).  The per element arithmetic does not
    depend on the tiling, so ``y`` and ``dx`` are bitwise the legacy plan's
    (``dweight``'s CTA rows follow the tiling).  ``legacy=True`` is the
    rule the package shipped before round 6, ``ct_vec = min(cv, threads)``,
    for either role; the KCP path keeps it (its kernels pick the grouped path
    from ``ct_vec`` on the device and are measured under this rule).  Each
    role's rule is kept in sync with its copies (the kernel generator of
    that role and the test plan of that kind).
    """

    return _channel_split(
        int(C), int(geometry["vec"]), int(geometry["threads"]), role, bool(legacy)
    )


@functools.lru_cache(maxsize=4096)
def _channel_split(
    C: int, vec: int, threads: int, role: str | None, legacy: bool
) -> tuple[int, int, int]:
    """:func:`channel_split` of plain arguments, memoized: a model's channel
    counts are few, so every call after a channel count's first reuses its
    plan (a backward call plans its row up to twice)."""

    if (role is None and not legacy) or (role is not None and role not in SPLIT_ROLES):
        raise ValueError(
            f"role must be one of {SPLIT_ROLES} (or legacy=True), got {role!r}"
        )
    if C <= 0 or C % vec:
        raise ValueError(f"C={C} must be a positive multiple of {vec}")
    channel_vectors = C // vec
    if legacy or channel_vectors <= threads:
        ct_vec = min(channel_vectors, threads)
    else:
        candidates = [
            width
            for width in range(32, threads + 1, 32)
            if channel_vectors % width == 0
        ]
        ct_vec = (
            max(candidates, key=lambda width: (width * (threads // width), width))
            if candidates
            else _fallback_split(channel_vectors, threads, role)
        )
    return ct_vec, threads // ct_vec, _ceil_div(channel_vectors, ct_vec)


def _fallback_split(channel_vectors: int, threads: int, role: str) -> int:
    """``ct_vec`` of :func:`channel_split` above ``threads`` vectors with no 32-multiple divisor.

    The candidates are every multiple of 32 at most ``threads`` (the last tile
    partial: its copies and stores stop at channel ``C``) and, for the
    forward, every divisor of ``channel_vectors`` from 40 to ``threads`` (whole
    tiles); the one of the highest :func:`_lane_use`, rounded to 9 decimals,
    wins, the larger on a tie.  No tile is narrower than a warp's 32 vectors
    and no whole forward tile narrower than 40.
    """

    candidates = set(range(32, threads + 1, 32))
    if role == "fwd":
        candidates |= {
            width for width in range(40, threads + 1) if channel_vectors % width == 0
        }
    return max(
        candidates,
        key=lambda width: (
            round(_lane_use(channel_vectors, width, threads), 9),
            width,
        ),
    )


def _lane_use(channel_vectors: int, width: int, threads: int) -> float:
    """The busy share of a CTA's lanes over ``ceil(channel_vectors / width)`` tiles of ``width``."""

    return (
        channel_vectors
        / (_ceil_div(channel_vectors, width) * width)
        * (width * (threads // width) / threads)
    )


def plan(
    S: int,
    C: int,
    num_docs: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
) -> dict[str, int]:
    """The kernel's launch scalars for one problem under a record's build geometry and role.

    ``geometry`` is ``record["grid_rule"]["geometry"]``: ``threads`` per CTA
    and ``vec`` BF16 channels per thread; ``role`` is ``record["role"]``
    (``"fwd"`` or ``"bwd"``).  The scalars are ``S``, ``C``, ``num_docs``,
    :func:`channel_split`'s ``ct_vec`` / ``n_sub`` / ``n_ch_tiles`` (``role``
    and ``legacy`` are passed on) and ``search_steps``, the number of halvings
    of the device-side binary search over ``cu_seqlens`` (``max(1, ceil(log2
    num_docs))``).  Every value is a pure function of ``(S, C, num_docs)``, the
    geometry and the role: the host never reads ``cu_seqlens``, and the kernel
    derives its token ranges from ``S`` and the grid.  The ``flags`` scalar of
    the ``dtype_flags`` programs is not a plan value: the host derives it from
    the tensors of the call.
    """

    S, C, num_docs = int(S), int(C), int(num_docs)
    if S < 0 or num_docs < 0:
        raise ValueError(f"S and num_docs must be non-negative, got {(S, num_docs)}")
    ct_vec, n_sub, n_ch_tiles = channel_split(
        C, geometry=geometry, role=role, legacy=legacy
    )
    return {
        "S": S,
        "C": C,
        "num_docs": num_docs,
        "search_steps": max(1, (num_docs - 1).bit_length()),
        "ct_vec": ct_vec,
        "n_sub": n_sub,
        "n_ch_tiles": n_ch_tiles,
    }


# Whole-rank trim of the backward's single-wave main grid (:func:`main_grid`).
# A channel tile ``ct`` of a ``grid``-CTA launch gets ``ceil((grid - ct) /
# n_ch_tiles)`` CTAs, so when ``n_ch_tiles`` does not divide the grid two pairs
# of neighbouring tiles stream other token ranges at any instant; a row that is
# not 64-byte aligned shares a cache line (and a 32-byte ``dx`` sector) between
# them, which beyond L2 is then fetched twice.  Dropping ``grid % n_ch_tiles``
# CTAs gives every tile the same CTA count.  ``TILE_ALIGNED_GRID[arch]``: the
# co-resident grid the rule was measured at (GB300's 304 CTAs; 0: none), on the
# routes of ``TILE_ALIGNED_ROUTES``, from ``TILE_ALIGNED_MIN_S`` tokens, with a
# last tile of at least ``TILE_ALIGNED_MIN_LAST[ct_vec]`` vectors and at most
# ``TILE_ALIGNED_MAX_TILES[ct_vec]`` tiles (a ``ct_vec`` without an entry:
# never).  ``TILE_ALIGNED_CLASSES[arch]`` holds the other devices' measured
# classes, ``(co_resident, ct_vec, n_ch_tiles) -> (min S, min last-tile
# vectors)`` on the routes of ``TILE_ALIGNED_CLASS_ROUTES[arch]`` (B200's 296
# CTAs; R200 none).  Every other launch keeps its grid.  ``dx`` does not depend
# on the grid; ``dweight`` / ``dbias`` sum other CTA rows.  The rule (the det
# package's, kept for the same bodies) is kept in sync with its copy in the
# backward kernel generator.
TILE_ALIGNED_GRID: dict[str, int] = {"sm_100a": 0, "sm_103a": 304, "sm_107a": 0}
TILE_ALIGNED_ROUTES: tuple[str, ...] = ("bwd",)
TILE_ALIGNED_MIN_S = 98304
TILE_ALIGNED_MIN_LAST: dict[int, int] = {32: 15, 64: 49}
TILE_ALIGNED_MAX_TILES: dict[int, int] = {32: 27, 64: 9}
TILE_ALIGNED_CLASSES: dict[str, dict[tuple[int, int, int], tuple[int, int]]] = {
    "sm_100a": {
        (296, 32, 5): (147456, 27),
        (296, 32, 7): (65536, 31),
        (296, 64, 3): (147456, 59),
    },
    "sm_103a": {},
    "sm_107a": {},
}
TILE_ALIGNED_CLASS_ROUTES: dict[str, tuple[str, ...]] = {
    "sm_100a": ("bwd", "bwd_dt", "bwd_fp32g"),
    "sm_103a": (),
    "sm_107a": (),
}


def main_grid(
    co_resident: int,
    *,
    arch: str,
    grid_mult: int = 1,
    C: int | None = None,
    n_ch_tiles: int = 1,
    S: int | None = None,
    ct_vec: int | None = None,
    route: str | None = None,
) -> int:
    """The backward main launch's CTA count: ``grid_mult x co_resident``.

    ``co_resident`` is the program's co-resident CTA count on the device
    (``ctas_per_sm x SM count``).  A single-wave launch (``grid_mult`` 1) of
    a row the caller names (``C``, ``S``, the plan's ``n_ch_tiles`` and
    ``ct_vec``, the ``route``) is trimmed to whole channel-tile ranks:
    :func:`tile_aligned_grid` on a route of ``TILE_ALIGNED_ROUTES`` when
    ``co_resident`` is ``TILE_ALIGNED_GRID[arch]``, else
    :func:`class_aligned_grid` on a route of
    ``TILE_ALIGNED_CLASS_ROUTES[arch]``.  A caller that names no row keeps the
    untrimmed grid, the most CTAs any row launches.
    """

    if type(grid_mult) is not int or grid_mult < 1:
        raise ValueError(f"grid_mult must be a positive int, got {grid_mult!r}")
    grid = int(co_resident) * grid_mult
    named = None not in (C, S, ct_vec)
    if (
        grid_mult == 1
        and co_resident == TILE_ALIGNED_GRID[arch]
        and route in TILE_ALIGNED_ROUTES
        and named
    ):
        grid = tile_aligned_grid(grid, C, n_ch_tiles, S=S, ct_vec=ct_vec)
    elif grid_mult == 1 and route in TILE_ALIGNED_CLASS_ROUTES[arch] and named:
        grid = class_aligned_grid(grid, C, n_ch_tiles, arch=arch, S=S, ct_vec=ct_vec)
    return grid


def tile_aligned_grid(
    grid: int, C: int, n_ch_tiles: int, *, S: int, ct_vec: int, vec: int = 8
) -> int:
    """``grid`` less ``grid % n_ch_tiles`` CTAs when ``C % 32`` (a row of
    ``x`` / ``dy`` / ``dx`` is not 64-byte aligned), ``1 < n_ch_tiles <=
    TILE_ALIGNED_MAX_TILES[ct_vec]``, ``grid % n_ch_tiles``, ``grid >
    n_ch_tiles``, ``S >= TILE_ALIGNED_MIN_S`` and the last channel tile holds
    at least ``TILE_ALIGNED_MIN_LAST[ct_vec]`` vectors (``C / vec -
    (n_ch_tiles - 1) x ct_vec``); else ``grid``."""

    last = C // vec - (n_ch_tiles - 1) * ct_vec
    if (
        1 < n_ch_tiles <= TILE_ALIGNED_MAX_TILES.get(ct_vec, 0)
        and C % 32
        and grid % n_ch_tiles
        and grid > n_ch_tiles
        and S >= TILE_ALIGNED_MIN_S
        and last >= TILE_ALIGNED_MIN_LAST.get(ct_vec, ct_vec + 1)
    ):
        return grid - grid % n_ch_tiles
    return grid


def class_aligned_grid(
    grid: int,
    C: int,
    n_ch_tiles: int,
    *,
    arch: str,
    S: int,
    ct_vec: int,
    vec: int = 8,
) -> int:
    """``grid`` less ``grid % n_ch_tiles`` CTAs when ``C % 32``, ``grid %
    n_ch_tiles`` and ``(grid, ct_vec, n_ch_tiles)`` is a class of
    ``TILE_ALIGNED_CLASSES[arch]`` whose bounds the row meets (``S`` and the
    last tile's ``C / vec - (n_ch_tiles - 1) x ct_vec`` vectors at least the
    class's); else ``grid``."""

    bound = TILE_ALIGNED_CLASSES[arch].get((grid, ct_vec, n_ch_tiles))
    if bound is None or not C % 32 or not grid % n_ch_tiles or bound[0] > S:
        return grid
    if C // vec - (n_ch_tiles - 1) * ct_vec < bound[1]:
        return grid
    return grid - grid % n_ch_tiles


def row_may_trim(
    record: dict[str, Any], arch: str, co_resident: int, factor: int, C: int, S: int
) -> bool:
    """Whether :func:`main_grid` may trim ``record``'s launch for the row.

    Every trim needs a single wave (``factor`` 1), ``C % 32``, the record on a
    route of the rule and ``S`` at least the rule's smallest token bound for
    ``co_resident``; a call failing any of these launches the untrimmed grid
    without planning the row's channel tiles (no host cost beyond the
    untrimmed grid's)."""

    if factor != 1 or not C % 32:
        return False
    library = record["library"]
    routes = ROUTES[arch]
    if (
        co_resident == TILE_ALIGNED_GRID[arch]
        and S >= TILE_ALIGNED_MIN_S
        and any(library in routes.get(kind, ()) for kind in TILE_ALIGNED_ROUTES)
    ):
        return True
    return any(
        grid == co_resident and bound[0] <= S
        for (grid, _, _), bound in TILE_ALIGNED_CLASSES[arch].items()
    ) and any(
        library in routes.get(kind, ()) for kind in TILE_ALIGNED_CLASS_ROUTES[arch]
    )


def record_route(record: dict[str, Any], arch: str) -> str:
    """The route of ``ROUTES[arch]`` that lists ``record`` (``ValueError`` if none)."""

    return _library_route(record["library"], arch)


@functools.cache
def _library_route(library: str, arch: str) -> str:
    for kind, names in ROUTES[arch].items():
        if library in names:
            return kind
    raise ValueError(f"module {library} is on no route of {arch}")


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any],
    *,
    device_index: int,
    C: int | None = None,
    S: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one module on ``device_index`` under its recorded grid rule.

    ``co_resident`` modules launch exactly ``ctas_per_sm x SM count`` CTAs,
    the co-resident count.  ``occupancy`` modules (the backward programs, every
    one an ordinary launch) launch :func:`main_grid` of that count:
    ``grid_mult[arch] x ctas_per_sm x SM count`` CTAs normally
    (:func:`grid_per_sm`): ``ctas_per_sm`` co-resident CTAs per SM and
    ``grid_mult`` times finer token streams, so a CTA that finds no free slot
    beside another stream's kernel trails by a short stream in a later wave.
    Given the call's ``C`` and ``S``, a single-wave launch of the rows
    :func:`main_grid` names (its route, the record's plan) drops CTAs to whole
    channel-tile ranks (the row is planned only where :func:`row_may_trim`
    holds).  Either grid is a pure function of ``(S, C, device)`` and the
    route; ``dx`` does not depend on it, and the accumulator workspace does
    not either (:func:`accumulator_layout`).
    """

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind in ("co_resident", "occupancy"):
        arch = device_arch(device_index)
        grid_x = grid_per_sm(record, arch) * sm_count(device_index)
        if kind == "occupancy" and C is not None and S is not None and C % 32:
            co_resident = int(rule["ctas_per_sm"]) * sm_count(device_index)
            factor = grid_mult(record, arch)
            if row_may_trim(record, arch, co_resident, factor, int(C), int(S)):
                ct_vec, _, n_ch_tiles = channel_split(
                    C, geometry=rule["geometry"], role=record["role"]
                )
                grid_x = main_grid(
                    co_resident,
                    arch=arch,
                    grid_mult=factor,
                    C=int(C),
                    n_ch_tiles=n_ch_tiles,
                    S=int(S),
                    ct_vec=ct_vec,
                    route=record_route(record, arch),
                )
    else:
        raise ValueError(
            f"unknown Cake causal conv1d training grid rule: {kind!r} (known: {GRID_KINDS})"
        )
    return (max(1, grid_x), 1, 1)


def _align(nbytes: int) -> int:
    """``nbytes`` rounded up to :data:`WORKSPACE_ALIGN`."""

    return -(-int(nbytes) // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


def _accumulator_rule(record: dict[str, Any]) -> dict[str, Any]:
    """The ``accumulators`` workspace rule of a backward record (``ValueError`` for any other record)."""

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "accumulators":
        raise ValueError(
            f"module {record['kernel']} owns no backward accumulator workspace"
        )
    if rule["acc"] not in ACCUMULATOR_DTYPES or rule["work"] not in COUNTER_DTYPES:
        raise ValueError(
            f"module {record['kernel']}: unknown accumulator dtypes {rule['acc']!r} / {rule['work']!r}"
        )
    slots = rule.get("slots", 1)
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise ValueError(
            f"module {record['kernel']}: the accumulator rule's slots must be a positive int, got {slots!r}"
        )
    return rule


def accumulator_words_per_channel(record: dict[str, Any]) -> int:
    """Accumulator elements per channel of a backward record's rule: ``slots x taps`` (``slots`` copies of the
    ``taps`` per-channel sums; 1 copy when the rule names none)."""

    rule = _accumulator_rule(record)
    return int(rule.get("slots", 1)) * int(rule["taps"])


def _accumulator_bytes_per_channel(record: dict[str, Any]) -> int:
    return accumulator_words_per_channel(record) * ACCUMULATOR_DTYPES[
        record["workspace_rule"]["acc"]
    ]


def accumulator_layout(
    record: dict[str, Any], C: int, n_ch_tiles: int
) -> dict[str, Any]:
    """Byte layout of one backward launch's ``acc`` / ``work`` inside the ``uint8`` backward workspace.

    ``acc``: ``slots x taps x C`` words of the rule's ``acc`` dtype (FP32, or
    FP64 on the ``_fp32g`` routes) at offset 0 -- ``slots`` copies (the rule's,
    1 when it names none; the program spreads its CTAs over them) of the
    ``[C, 4]`` tap accumulators, then the ``[C]`` bias accumulators on the bias
    routes, in the program's own order; ``work``: ``n_ch_tiles``
    uint32 ticket counters at the next :data:`WORKSPACE_ALIGN` boundary.  The
    kernels leave both zero after every launch, so a zeroed buffer stays zero.
    The layout depends on ``C``, the plan's ``n_ch_tiles`` and the route,
    never on ``S`` or the launch grid.
    """

    rule = _accumulator_rule(record)
    C, n_ch_tiles = int(C), int(n_ch_tiles)
    if C <= 0 or n_ch_tiles <= 0:
        raise ValueError(f"C and n_ch_tiles must be positive, got {(C, n_ch_tiles)}")
    acc_elems = accumulator_words_per_channel(record) * C
    acc_bytes = acc_elems * ACCUMULATOR_DTYPES[rule["acc"]]
    work_offset = _align(acc_bytes)
    work_bytes = n_ch_tiles * COUNTER_DTYPES[rule["work"]]
    return {
        "acc_dtype": rule["acc"],
        "acc_elems": acc_elems,
        "acc_bytes": acc_bytes,
        "work_dtype": rule["work"],
        "work_offset": work_offset,
        "work_words": n_ch_tiles,
        "total_bytes": _align(work_offset + work_bytes),
    }


def backward_workspace_bytes(arch: str, C: int) -> int:
    """Bytes of the public backward workspace for ``C`` channels on ``arch``.

    The largest :func:`accumulator_layout` over every backward program of every
    backward route of the architecture (each planned with its own geometry and
    role), so one zeroed buffer serves whichever program a call selects.  It
    depends on ``C`` and the architecture only (not on ``S``, the SM count or
    the grid).
    """

    nbytes = 0
    for owner in workspace_owners(arch):
        _, _, n_ch_tiles = channel_split(
            C, geometry=owner["grid_rule"]["geometry"], role=owner["role"]
        )
        nbytes = max(
            nbytes, int(accumulator_layout(owner, C, n_ch_tiles)["total_bytes"])
        )
    return nbytes


def counter_words(record: dict[str, Any], n_ch_tiles: int) -> int:
    """Int32 words of a forward record's ``work`` counters for a launch of ``n_ch_tiles`` channel tiles.

    A ``tile_counters`` rule holds ``words_per_tile`` words per channel tile
    (the dynamic forward's next-chunk counter and done-CTA counter); a record
    without a workspace rule takes no ``work`` (0 words).
    """

    rule = record["workspace_rule"]
    if rule is None:
        return 0
    if rule["kind"] != "tile_counters":
        raise ValueError(
            f"module {record['kernel']}: workspace rule {rule['kind']!r} holds no forward counters"
        )
    if n_ch_tiles <= 0:
        raise ValueError(f"n_ch_tiles must be positive, got {n_ch_tiles}")
    return int(rule["words_per_tile"]) * int(n_ch_tiles)


def forward_kinds(arch: str) -> tuple[str, ...]:
    """The forward routes ``arch`` exports, in :data:`KINDS` order."""

    return tuple(
        kind
        for kind in KINDS
        if route_role(kind) == "fwd" and ROUTES.get(arch, {}).get(kind)
    )


def forward_workspace_words(arch: str, C: int) -> int:
    """Int32 words of the forward workspace for ``C`` channels on ``arch``.

    The largest :func:`counter_words` over every forward program of every
    forward route of the architecture (each planned with its own geometry and
    role), so one buffer serves every forward route; it depends on ``C`` only
    (never on ``S`` or the device's SM count).  0 when no forward program of
    the architecture takes ``work``.
    """

    words = 0
    for kind in forward_kinds(arch):
        for name in route_modules(arch, kind):
            record = MODULES[name]
            if record["workspace_rule"] is None:
                continue
            _, _, n_ch_tiles = channel_split(
                C, geometry=record["grid_rule"]["geometry"], role=record["role"]
            )
            words = max(words, counter_words(record, n_ch_tiles))
    return words


def is_rendered() -> bool:
    """Whether the exporter has rendered this package: its ``build.json`` exists next to this file."""

    return (Path(__file__).parent / "build.json").is_file()


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake causal conv1d training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake causal conv1d training module {name} is not delivered for {arch}; "
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


_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


@functools.cache
def _arg_sources(name: str) -> tuple[tuple[bool, Any], ...]:
    """``(grid, source)`` per entry of ``MODULES[name]["arg_plan"]``, in
    order: a grid slot (``source`` is the axis, 0 / 1 / 2 for ``grid_x`` /
    ``grid_y`` / ``grid_z``) or the logical name the kernel parameter binds
    (the record's ``logical`` map); the registry is constant, so
    :func:`run_modules` reads it once per module."""

    record = MODULES[name]
    logical = record["logical"]
    return tuple(
        (True, _GRID_AXES[key]) if kind == "grid" else (False, logical[key])
        for kind, key in record["arg_plan"]
    )


def _missing_argument(name: str, values: dict[str, Any]) -> ValueError:
    """The error of a launch of module ``name`` whose ``values`` lack a
    logical name it binds (the first such name, with its kernel parameter)."""

    record = MODULES[name]
    logical, key = next(
        (record["logical"][key], key)
        for kind, key in record["arg_plan"]
        if kind != "grid" and record["logical"][key] not in values
    )
    return ValueError(f"module {name} needs {logical!r} (kernel parameter {key!r})")


@functools.cache
def _ffi_entry(name: str, device_index: int) -> Any:
    """The FFI entry of module ``name`` loaded for CUDA device
    ``device_index``'s architecture (:func:`load`), looked up once."""

    module = load(name, device_arch(device_index))
    return getattr(module, MODULES[name]["ffi_entry"])


@functools.cache
def _raw_stream_getter() -> Any:
    """``torch._C._cuda_getCurrentRawStream``: the handle of a device's
    current torch stream as an integer, without constructing a ``Stream``
    object; or the ``torch.cuda`` form on a torch build without it.  Bound on
    first use: this module imports torch lazily."""

    import torch

    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is not None:
        return getter

    def fallback(device_index: int) -> int:
        return torch.cuda.current_stream(device_index).cuda_stream

    return fallback


_FFI_DEVICES: dict[int, Any] = {}


def _stream_context(device_index: int) -> Any:
    """The tvm-ffi environment-stream context of torch's current stream on
    CUDA device ``device_index`` (a CUDA graph's capture stream during
    capture): the bindings launch on that stream.  The FFI device handle is
    kept per index, so entering the context costs a fraction of
    ``tvm_ffi.use_torch_stream()``, which builds a ``torch.cuda.Stream`` and
    parses a device string on every entry."""

    device = _FFI_DEVICES.get(device_index)
    if device is None:
        device = _FFI_DEVICES[device_index] = tvm_ffi.device(f"cuda:{device_index}")
    return tvm_ffi.use_raw_stream(device, _raw_stream_getter()(device_index))


def run_modules(
    launches: Sequence[tuple[str, tuple[int, int, int] | None]],
    values: dict[str, Any],
    *,
    device_index: int,
) -> None:
    """Launch exported modules in order, back to back, with one set of logical
    argument ``values``.

    ``launches`` holds ``(name, grid)`` pairs, ``grid`` as in
    :func:`run_module`.  All of them are issued inside one stream context of
    torch's current stream on ``device_index``, each bound (grid and
    arguments) right before it is issued, so the first launch waits for no
    later one's binding.  ``ValueError`` names the first logical
    name a module binds that ``values`` lacks.
    """

    with _stream_context(device_index):
        for name, grid in launches:
            if grid is None:
                grid = launch_grid(
                    MODULES[name],
                    device_index=device_index,
                    C=values.get("C"),
                    S=values.get("S"),
                )
            sources = _arg_sources(name)
            try:
                args = [
                    grid[source] if is_grid else values[source]
                    for is_grid, source in sources
                ]
            except KeyError:
                raise _missing_argument(name, values) from None
            _ffi_entry(name, device_index)(*args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    device_index: int,
    grid: tuple[int, int, int] | None = None,
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is
    ``grid`` when the caller planned it (the module's :func:`launch_grid` for
    the call, as the public entry points pass it), else the module's
    :func:`launch_grid` on ``device_index`` for the call's ``C`` and ``S``,
    and the kernel runs on the current torch stream; a
    module whose launch record enables programmatic dependent launch
    (``use_pdl``) overlaps its launch with the preceding kernel of that stream
    and waits for it in-kernel.
    """

    run_modules(((name, grid),), values, device_index=device_index)


__all__ = [
    "ACCUMULATOR_DTYPES",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHANNEL_MULTIPLE",
    "COUNTER_DTYPES",
    "FLAG_B_BF16",
    "FLAG_DB_BF16",
    "FLAG_DW_BF16",
    "FLAG_W_BF16",
    "GRID_KINDS",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PLAN_SCALARS",
    "REQUIRED_KINDS",
    "ROLES",
    "ROUTES",
    "SELECT_KINDS",
    "TILE_ALIGNED_CLASSES",
    "TILE_ALIGNED_CLASS_ROUTES",
    "TILE_ALIGNED_GRID",
    "TILE_ALIGNED_MAX_TILES",
    "TILE_ALIGNED_MIN_LAST",
    "TILE_ALIGNED_MIN_S",
    "TILE_ALIGNED_ROUTES",
    "WIDTH",
    "WORKSPACE_ALIGN",
    "WORKSPACE_KINDS",
    "accumulator_layout",
    "accumulator_words_per_channel",
    "arch_for_capability",
    "backward_kinds",
    "backward_route",
    "backward_workspace_bytes",
    "channel_split",
    "class_aligned_grid",
    "counter_words",
    "device_arch",
    "forward_kinds",
    "forward_route",
    "forward_workspace_words",
    "grid_mult",
    "grid_per_sm",
    "is_rendered",
    "launch_grid",
    "library_name",
    "load",
    "main_grid",
    "plan",
    "record_route",
    "route_applies",
    "route_modules",
    "route_role",
    "row_may_trim",
    "run_module",
    "run_modules",
    "select_rule",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "tile_aligned_grid",
    "workspace_owners",
    "workspace_record",
]
