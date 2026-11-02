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
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from . import cake_build

# Registry of the generated ragged grouped GEMM programs of the
# non-deterministic package.  ``PROGRAMS`` and ``ROUTES`` are written by the
# Cake generated-program export; the host plan constants and the functions
# below are this package's.
#
# ``PROGRAMS``: one record per physical program -- one kernel source plus its
# host binding, shared by every architecture in ``arches`` (genuine
# per-architecture lowering differences live inside the source under exact
# ``__CUDA_ARCH__`` regions) -- with its architecture-free ``library`` stem, FFI
# entry, argument plan and launch shape.  What to compile (the translation
# units, the compile flags, the per-architecture closure digest) is recorded in
# ``build.json`` (see ``cake_build``), whose entry for a program is
# ``<library>[<specialization tag>]_<arch>``; a program is built once per
# (architecture, compile-line specialization), prebuilt under ``aot/`` or by
# FlashInfer's JIT.
#
# ``ROUTES``: one record per public route -- ``fwd``, ``dgrad`` and
# ``wgrad_{bf16,f32}_k{256,512}`` -- naming its architectures and ordered
# stages.  Each stage binds a program and the compile-line specializations the
# loader defines for it (``WGRAD_OUT`` = 16 for bf16 or 32 for fp32 weight
# gradients, from the route's output dtype; ``WGRAD_TILE``, the k width of the
# partials the tail-reduce program sums, from the route's tile).  A stage
# serves the architectures its program is built for among the route's: a route
# may name two stages of one name whose programs serve disjoint architectures,
# so a stage is looked up per architecture (``stage_binding``).
#
# The forward, data-gradient and 256-wide k tile weight-gradient routes bind
# this package's copies of the deterministic package's programs
# (``flashinfer.experimental.cake_moe_grouped_det_gemm``: the same builds of the
# same kernels, so the same output bits; the 256-wide k tile's ``main_wave``
# program is the same kernel rendered from a newer source, with two more buffer
# arguments its sm_100a / sm_103a builds never read and no sm_107a build) with
# the same stages: ``main``,
# ``main_wide``, ``main_wide_ef`` and ``main_wide_ts33`` of ``fwd`` / ``dgrad``,
# and ``main``, ``main_wave`` and the ordered fp32 ``tail_reduce`` of the
# 256-wide k tile.  The 512-wide k tile's weight-gradient routes bind the
# builds of the deterministic package's k512 forms -- ``main``, ``main_wave``,
# ``main_halves`` (with the planned tile walk; on sm_107a with the slack tail
# placement too), ``main_wave_halves`` (waves synchronised every eight) and, on
# sm_107a, ``main_halves_k256t`` (with the slack tail placement) -- whose split
# tail is added in place: every unit of a split tail tile adds its fp32
# accumulator into the tile's fp32 sum in ``tail_sums`` with unordered
# reductions and counts its arrival on the tile's tickets in ``tail_tickets``;
# the last unit rounds the sum once, stores it and clears the sum and the
# tickets.  Those routes have no ``tail_reduce`` stage and launch one kernel.
# Every whole tile is reduced in a fixed order by one CTA pair, as in the
# deterministic package; only a split tail tile's units are added in an order
# the launch does not fix.  With the slack tail placement the kernel plans each
# launch's tail on the device: when every tail tile fits in the room the full
# waves leave on the least-loaded clusters, each runs whole on one of them (no
# sum or ticket is touched), else the tail splits as above.
#
# The host chooses among the stages a route serves by ``HOST_PLAN_CONSTANTS``,
# with the deterministic package's rules and constants (``cake_backend``,
# program choice); ``unordered_tail_archs`` lists the architectures whose
# 512-wide k tile routes bind the unordered-tail builds -- every architecture of
# those routes, since the registry holds no ordered k512 build (leaving one out
# needs the deterministic k512 builds and a ``tail_reduce`` stage on those routes,
# a new render) -- and ``atomic_tail_plan`` holds the cost-model constants of the
# unordered split tail (``tail_plan`` stays the ordered tail's).
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_037c741e69a0ccc696e0": {
        "library": "cake_moe_grouped_gemm_037c741e69a0ccc696e0",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_1edcad995dcc790600cb": {
        "library": "cake_moe_grouped_gemm_1edcad995dcc790600cb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb": {
        "library": "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_3edd5c53a1269a6095a7": {
        "library": "cake_moe_grouped_gemm_3edd5c53a1269a6095a7",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_3f70fc4be03d3507175c": {
        "library": "cake_moe_grouped_gemm_3f70fc4be03d3507175c",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_4f40b5f07baf6e086fdd": {
        "library": "cake_moe_grouped_gemm_4f40b5f07baf6e086fdd",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_5064b19de3716266ed03": {
        "library": "cake_moe_grouped_gemm_5064b19de3716266ed03",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_55833be10c8515562245": {
        "library": "cake_moe_grouped_gemm_55833be10c8515562245",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_594235460aac7658daab": {
        "library": "cake_moe_grouped_gemm_594235460aac7658daab",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_6bd667b99e236d0c69df": {
        "library": "cake_moe_grouped_gemm_6bd667b99e236d0c69df",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_6be356d4d7ce990aa413": {
        "library": "cake_moe_grouped_gemm_6be356d4d7ce990aa413",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_6db26c5927b4738a8b8f": {
        "library": "cake_moe_grouped_gemm_6db26c5927b4738a8b8f",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_7e39db00fc7c99693f3e": {
        "library": "cake_moe_grouped_gemm_7e39db00fc7c99693f3e",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_994f65a456ae68a92e84": {
        "library": "cake_moe_grouped_gemm_994f65a456ae68a92e84",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
    },
    "cake_moe_grouped_gemm_9df3c426f0afac83124c": {
        "library": "cake_moe_grouped_gemm_9df3c426f0afac83124c",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_aba5ff762eb9355f540a": {
        "library": "cake_moe_grouped_gemm_aba5ff762eb9355f540a",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c": {
        "library": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["parameter", "num_clusters"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_c6cc9d6508fe7a5e9b67": {
        "library": "cake_moe_grouped_gemm_c6cc9d6508fe7a5e9b67",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_ce655879551e583539e2": {
        "library": "cake_moe_grouped_gemm_ce655879551e583539e2",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
    },
    "cake_moe_grouped_gemm_ed9a74f36ba13a52183d": {
        "library": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb": {
        "library": "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_dgrad": {
        "op": "dgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_7e39db00fc7c99693f3e",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_1edcad995dcc790600cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_6bd667b99e236d0c69df",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_aba5ff762eb9355f540a",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_5064b19de3716266ed03",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_gemm_994f65a456ae68a92e84",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_fwd": {
        "op": "fwd",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_9df3c426f0afac83124c",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_55833be10c8515562245",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_3edd5c53a1269a6095a7",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_4f40b5f07baf6e086fdd",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_gemm_ce655879551e583539e2",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k256": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k512": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_c6cc9d6508fe7a5e9b67",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_6db26c5927b4738a8b8f",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_6be356d4d7ce990aa413",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_3f70fc4be03d3507175c",
                "specializations": {},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_gemm_037c741e69a0ccc696e0",
                "specializations": {},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_gemm_594235460aac7658daab",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_f32_k256": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_f32_k512": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_c6cc9d6508fe7a5e9b67",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_6db26c5927b4738a8b8f",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_6be356d4d7ce990aa413",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_3f70fc4be03d3507175c",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_gemm_037c741e69a0ccc696e0",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_gemm_594235460aac7658daab",
                "specializations": {"WGRAD_OUT": 32},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
}
HOST_PLAN_CONSTANTS: dict[str, Any] = {
    "block_n": 256,
    "wide_block_n": 512,
    "wide_fwd_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "wide_dgrad_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "dgrad_evict_first_archs": ["sm_107a"],
    "dgrad_evict_first_min_rows": 16384,
    "dgrad_evict_first_min_k": 4096,
    "fwd_lwh_ef_archs": ["sm_107a"],
    "fwd_lwh_ef_max_k": 2048,
    "dgrad_wide_ts33_archs": ["sm_103a"],
    "dgrad_wide_ts33_min_rows": {"sm_103a": 8192},
    "dgrad_wide_ts33_min_k": 4096,
    "fwd_wide_ts33_archs": ["sm_103a"],
    "fwd_wide_ts33_max_k": 2048,
    "fwd_wide_ts33_max_rows": 4096,
    "block_k": 64,
    "cta_group": 2,
    "cluster_m": 256,
    "epi_chunk": 16,
    "tmap_slot_bytes": 128,
    "tmap_slots_per_cta": 18,
    "k512_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "k512_release_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "k256_tail_archs": ["sm_107a"],
    "k256_tail_max_k": 2048,
    "k256_tail_min_rows": 8192,
    "k512_min_rows_per_group": {"sm_100a": 4096, "sm_103a": 4096, "sm_107a": 4096},
    "wave_sync_min_k": 4096,
    "wave_sync_archs": {
        256: ["sm_100a", "sm_103a"],
        512: ["sm_100a", "sm_103a"],
    },
    "wave_sync_mode": {"sm_100a": 2, "sm_103a": 2},
    "wave_sync_words": 2,
    "sched_words": 4224,
    "sched_rest": [0, 2147483648, 0],
    "domain_split_archs": ["sm_100a", "sm_103a"],
    "domain_split_ops": ["dgrad"],
    "domain_split_max_weight_elems": 12582912,
    "wgrad_tail_split": 1,
    "tail_plan": {
        "max_splits": 16,
        "sm_clock_ghz": 1.8,
        "dram_tbps": 6.0,
        "launch_us": 5.0,
        "cycles_per_step": {256: 512, 512: 1024},
    },
    # The architectures whose 512-wide k tile routes bind the unordered-tail
    # builds (``cake_backend.wgrad_unordered_tail``).
    "unordered_tail_archs": ["sm_100a", "sm_103a", "sm_107a"],
    # The split-tail cost model of the 512-wide k tile's unordered tail
    # (``cake_backend.wgrad_atomic_tail_plan``): the keys of ``tail_plan``.
    "atomic_tail_plan": {
        "max_splits": 16,
        "sm_clock_ghz": 1.8,
        "dram_tbps": 6.0,
        "launch_us": 5.0,
        "cycles_per_step": {256: 512, 512: 1024},
    },
    "int32_elems": 2147483647,
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
OPS = ("fwd", "dgrad", "wgrad")


def record_name(
    op: str, *, out_dtype: Optional[str] = None, tile_k: Optional[int] = None
) -> str:
    """Route key of ``op``; weight-gradient routes are further keyed by output dtype and k tile."""
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    base = f"cake_moe_grouped_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient routes are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return base


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> bool:
    """True when this checkout registers a route of ``op`` for ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.
    """
    if arch not in ARCHES:
        return False
    if op != "wgrad":
        route = ROUTES.get(record_name(op))
        return route is not None and arch in route["arches"]
    dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
    tiles = (tile_k,) if tile_k else (256, 512)
    for dtype in dtypes:
        for tile in tiles:
            route = ROUTES.get(record_name(op, out_dtype=dtype, tile_k=tile))
            if route is not None and arch in route["arches"]:
                return True
    return False


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered route name serving ``op`` on ``arch`` or raise."""
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    name = record_name(op, out_dtype=out_dtype, tile_k=tile_k)
    route = ROUTES.get(name)
    if route is None or arch not in route["arches"]:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM route {name!r} is not registered for "
            f"{arch} in this checkout (the registry of "
            "flashinfer.experimental.cake.cake_moe_grouped_gemm.cake_jit is filled by the "
            "generated-program export)"
        )
    if route["op"] != op:
        raise RuntimeError(
            f"registered route {name!r} is a {route['op']} route, bound to {op}"
        )
    return name


def stage_binding(name: str, stage: str, arch: str) -> dict[str, Any]:
    """The ``{"name", "program", "specializations"}`` record of ``stage`` of route ``name`` on ``arch``."""
    route = ROUTES[name]
    for item in route["stages"]:
        if item["name"] == stage and arch in PROGRAMS[item["program"]]["arches"]:
            return item
    raise KeyError(f"route {name!r} has no {stage!r} stage on {arch}")


def stage_served(name: str, stage: str, arch: str) -> bool:
    """True when route ``name`` has a ``stage`` stage whose program is built for ``arch``."""
    return any(
        item["name"] == stage and arch in PROGRAMS[item["program"]]["arches"]
        for item in ROUTES[name]["stages"]
    )


def _specialization_items(
    specializations: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(specializations.items()))


def specialization_tag(specializations: tuple[tuple[str, Any], ...]) -> str:
    """Library-name suffix of a compile-line specialization set: ``_wgrad_tile256`` for
    ``{"WGRAD_TILE": 256}``, empty for none.  ``build.json`` names a specialized build
    ``<library><tag>_<arch>`` and carries its ``-D<KEY>=<value>`` defines in that entry's ``nvcc_flags``."""
    return "".join(f"_{key.lower()}{value}" for key, value in specializations)


def library_name(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
) -> str:
    """``build.json`` entry of ``program`` built for ``arch`` under ``specializations``."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not delivered for {arch}")
    tag = specialization_tag(_specialization_items(specializations or {}))
    return f"{record['library']}{tag}_{arch}"


@functools.cache
def _load_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    return cake_build.load_library(library_name(program, arch, dict(specializations)))


def load_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """Load (once per program, exact target and specialization) one physical program: the prebuilt
    library under ``aot/`` when the package ships it, else FlashInfer's JIT build of its ``build.json``
    entry."""
    return _load_module(program, arch, _specialization_items(specializations or {}))
