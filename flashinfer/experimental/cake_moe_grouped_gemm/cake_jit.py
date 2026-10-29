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

# Registry of the generated ragged grouped GEMM programs.  ``PROGRAMS`` and
# ``ROUTES`` are written by the Cake generated-program export; the host plan
# constants and the functions below are this package's.
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
# ``wgrad_{bf16,f32}_k{256,512}`` -- naming its architectures and
# ordered stages.  Each stage binds a program and the compile-line
# specializations the loader defines for it (the weight-gradient tail-reduce
# program takes the k width of the partials it sums, ``WGRAD_TILE``, from the
# route's tile instead of carrying one source per tile; the weight-gradient
# programs take their output element width, ``WGRAD_OUT`` = 16 for bf16 or 32
# for fp32, from the route's output dtype instead of carrying one source per
# output dtype).  A stage serves the architectures its program is built for
# among the route's: a route may name two stages of one name whose programs
# serve disjoint architectures (the forward and data-gradient ``main`` and
# ``main_wide`` stages: the domain-scheduled program on sm_100a / sm_103a, the
# static program on sm_107a, whose data-gradient ``main_wide`` program keeps
# each G tile in the tensor core's A collector for both 256-column halves of
# the output tile; the 512-wide k tile's weight-gradient ``main`` stage: one
# build on sm_100a / sm_103a, another on sm_107a), so a stage is looked up per
# architecture (``stage_binding``).  The host chooses among the stages a route
# serves by ``HOST_PLAN_CONSTANTS``: the forward and data-gradient
# ``main_wide`` stage is the program with the 512-column output tile
# (``wide_block_n``), launched on the architectures of ``wide_fwd_archs`` /
# ``wide_dgrad_archs`` when the output columns are a multiple of it; the
# data-gradient ``main_wide_ef`` stage is its twin that loads G and stores dX
# with the L2 evict-first policy, so the group's weights stay resident in L2,
# launched in its place on the architectures of ``dgrad_evict_first_archs``
# when the groups average at least ``dgrad_evict_first_min_rows`` rows and the
# output has at least ``dgrad_evict_first_min_k`` columns; the forward
# ``main_wide_ef`` stage is its twin that releases the last wave of tiles in
# halves and stores Y with the L2 evict-first policy, launched in its place on
# the architectures of ``fwd_lwh_ef_archs`` when the reduction K is at most
# ``fwd_lwh_ef_max_k`` (the down projection); both ``main_wide_ef`` programs
# split two head and two tail k steps of an item around the release of the
# accumulator's halves instead of four and three; the weight gradient
# takes the 512-wide k tile on the architectures of ``k512_archs`` when the
# groups average at least the architecture's ``k512_min_rows_per_group`` rows;
# the weight-gradient ``main_wave`` stage is the main program with wave
# synchronisation, launched from ``wave_sync_min_k`` on the architectures
# ``wave_sync_archs`` lists for the k tile, with the architecture's
# ``wave_sync_mode``; the 512-wide k tile's ``main_halves`` and
# ``main_wave_halves`` stages are the twins of ``main`` and ``main_wave`` that
# release the accumulator in two halves, launched on the architectures of
# ``k512_release_archs``; its ``main_halves_k256t`` stage is the twin of
# ``main_halves`` that runs the tiles left after the full waves as 256-column
# half tiles, whose split tail the tail-reduce stage of the 256-wide k tile's
# route sums, launched in its place on the architectures of
# ``k256_tail_archs`` when K is at most ``k256_tail_max_k`` and the groups
# average at least ``k256_tail_min_rows`` rows.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_05258dbe307cb47d829e": {
        "library": "cake_moe_grouped_gemm_05258dbe307cb47d829e",
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
    "cake_moe_grouped_gemm_0669888e0651eb5ac046": {
        "library": "cake_moe_grouped_gemm_0669888e0651eb5ac046",
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
    "cake_moe_grouped_gemm_4af5480da78305cbf114": {
        "library": "cake_moe_grouped_gemm_4af5480da78305cbf114",
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
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_611af0ca12365b0e7c3b": {
        "library": "cake_moe_grouped_gemm_611af0ca12365b0e7c3b",
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
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_611f42e92c0a16307151": {
        "library": "cake_moe_grouped_gemm_611f42e92c0a16307151",
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
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_91100899e34d05c9b2bb": {
        "library": "cake_moe_grouped_gemm_91100899e34d05c9b2bb",
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
    "cake_moe_grouped_gemm_9140b3452aadfce554d6": {
        "library": "cake_moe_grouped_gemm_9140b3452aadfce554d6",
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
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_920598481c6dfe1ac044": {
        "library": "cake_moe_grouped_gemm_920598481c6dfe1ac044",
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
    "cake_moe_grouped_gemm_96f299e996aea2331023": {
        "library": "cake_moe_grouped_gemm_96f299e996aea2331023",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_9cd96cdceb35c7ac936e": {
        "library": "cake_moe_grouped_gemm_9cd96cdceb35c7ac936e",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_a30ea3df10b8ee9976f2": {
        "library": "cake_moe_grouped_gemm_a30ea3df10b8ee9976f2",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_aafe5d56af3c36f3b143": {
        "library": "cake_moe_grouped_gemm_aafe5d56af3c36f3b143",
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
    "cake_moe_grouped_gemm_ab84c164aef14d967ac3": {
        "library": "cake_moe_grouped_gemm_ab84c164aef14d967ac3",
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
    "cake_moe_grouped_gemm_bb31cf89a1f6335a7baa": {
        "library": "cake_moe_grouped_gemm_bb31cf89a1f6335a7baa",
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
    "cake_moe_grouped_gemm_ccb57e5ab20e09ac2f93": {
        "library": "cake_moe_grouped_gemm_ccb57e5ab20e09ac2f93",
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
    "cake_moe_grouped_gemm_f6441c89bf898d4db030": {
        "library": "cake_moe_grouped_gemm_f6441c89bf898d4db030",
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
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_dgrad": {
        "op": "dgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_bb31cf89a1f6335a7baa",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_1edcad995dcc790600cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_ab84c164aef14d967ac3",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_05258dbe307cb47d829e",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_ccb57e5ab20e09ac2f93",
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
                "program": "cake_moe_grouped_gemm_aafe5d56af3c36f3b143",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_0669888e0651eb5ac046",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_91100899e34d05c9b2bb",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_920598481c6dfe1ac044",
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
                "program": "cake_moe_grouped_gemm_9cd96cdceb35c7ac936e",
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
                "program": "cake_moe_grouped_gemm_611af0ca12365b0e7c3b",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_611f42e92c0a16307151",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_a30ea3df10b8ee9976f2",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_9140b3452aadfce554d6",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_4af5480da78305cbf114",
                "specializations": {},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_gemm_96f299e996aea2331023",
                "specializations": {},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_gemm_f6441c89bf898d4db030",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 512},
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
                "program": "cake_moe_grouped_gemm_9cd96cdceb35c7ac936e",
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
                "program": "cake_moe_grouped_gemm_611af0ca12365b0e7c3b",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_611f42e92c0a16307151",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_a30ea3df10b8ee9976f2",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_9140b3452aadfce554d6",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_4af5480da78305cbf114",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_gemm_96f299e996aea2331023",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_gemm_f6441c89bf898d4db030",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 512},
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
    "wgrad_tail_split": 1,
    "tail_plan": {
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
            "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit is filled by the "
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
