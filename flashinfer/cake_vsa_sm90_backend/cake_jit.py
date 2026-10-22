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

Host plan and loader glue of the Cake Hopper (SM90) variable block-sparse
attention kernels (BF16 HND, head_dim 128, 64-token blocks).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture (``sm_90a``).  ``MODULES`` carries the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch
record, route, grid rule, workspace rule and the kernel-parameter ->
logical-argument map.  What to compile -- sources, digests, nvcc flags -- is
recorded once in ``build.json`` next to this file; :mod:`.cake_build` serves
the library ``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder
compiled it and otherwise through FlashInfer's JIT from the same sources and
flags, verifying every source digest before the first build.

A record is one *stage*: the persistent kernel (``attention``, its
queue-scheduled twin ``attention_queue``, its compact static form
``attention_compact``) or one variant of the small-selection
kernel (``small_k<kmax>[s][c<cluster>]``: ``kmax`` selected KV blocks per CTA,
``s`` the split-KV variant with a plan-owned workspace, ``c<n>`` the
thread-block-cluster variant, ``sc<n>`` the split-cluster variant whose slices
form one cluster and merge through the workspace).  Which stage a problem runs is decided by the
host planner (:mod:`.cake_planner`, ``small_route`` / ``plan_small`` /
``plan_vsa_sm90``); the plan object (:class:`..CakeVsaSm90Plan`) binds the
stage's arguments once and launches it on every call.  Problem sizes are plan
data and runtime scalars, so a changed mask or sequence length never
recompiles anything.

The program name ``cake_vsa_sm90_<20 hex>`` is the first twenty hex digits of
the SHA-256 of the compact JSON ``{"build": <build keywords, sorted>,
"module": <dotted kernel module>}`` of the kernel the program was rendered
from: the persistent kernel with its default keywords and with
``queue=True``, the small kernel with ``kmax`` / ``split`` / ``cluster``.

Numerics: every ``build.json`` entry records ``--use_fast_math`` (the flag the
kernels were validated and measured with); the loader passes
``use_fast_math=False`` so that the manifest, not the loader, states it, and
no ``--fmad`` flag is recorded (nvcc's default contraction applies).
"""

from __future__ import annotations

import functools
from typing import Any, Mapping, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory and must
# equal it exactly: the exporter refuses a kernel that needs more and the
# launch gate refuses a record that states another value.
MODULES: dict[str, dict[str, Any]] = {
    "attention": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["buffer", "meta"],
            ["param_array", "hdr"],
            ["parameter", "tile_stride"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "dbg"],
            ["buffer", "tl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_ctas",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_746d3c6c7aed747d355b",
        "kernel_symbol": "kernel_cake_vsa_sm90_746d3c6c7aed747d355b",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232064,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_746d3c6c7aed747d355b",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "dbg": "dbg",
            "hdr": "hdr",
            "meta": "meta",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "tile_stride": "tile_stride",
            "tl": "tl",
        },
        "route": {
            "compact": False,
            "kind": "persistent",
            "queue": False,
            "tagged": False,
        },
        "workspace_rule": {
            "dbg_words": 64,
            "kind": "persistent",
            "tl_words": 8,
        },
    },
    "attention_queue": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["buffer", "meta"],
            ["param_array", "hdr"],
            ["parameter", "tile_stride"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "dbg"],
            ["buffer", "tl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_ctas",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_3c3e392830e702a1d2b1",
        "kernel_symbol": "kernel_cake_vsa_sm90_3c3e392830e702a1d2b1",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232192,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_3c3e392830e702a1d2b1",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "dbg": "dbg",
            "hdr": "hdr",
            "meta": "meta",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "tile_stride": "tile_stride",
            "tl": "tl",
        },
        "route": {
            "compact": False,
            "kind": "persistent",
            "queue": True,
            "tagged": False,
        },
        "workspace_rule": {
            "dbg_words": 64,
            "kind": "persistent",
            "tl_words": 8,
        },
    },
    "attention_compact": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["buffer", "meta"],
            ["param_array", "hdr"],
            ["parameter", "tile_stride"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "dbg"],
            ["buffer", "tl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_ctas",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_0f730e6ab6a65c1e7199",
        "kernel_symbol": "kernel_cake_vsa_sm90_0f730e6ab6a65c1e7199",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232064,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_0f730e6ab6a65c1e7199",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "dbg": "dbg",
            "hdr": "hdr",
            "meta": "meta",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "tile_stride": "tile_stride",
            "tl": "tl",
        },
        "route": {
            "compact": True,
            "kind": "persistent",
            "queue": False,
            "tagged": False,
        },
        "workspace_rule": {
            "dbg_words": 64,
            "kind": "persistent",
            "tl_words": 8,
        },
    },
    "attention_tagged": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["buffer", "meta"],
            ["param_array", "hdr"],
            ["parameter", "tile_stride"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "dbg"],
            ["buffer", "tl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_ctas",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_dead614631d8a6b8d69e",
        "kernel_symbol": "kernel_cake_vsa_sm90_dead614631d8a6b8d69e",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232064,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_dead614631d8a6b8d69e",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "dbg": "dbg",
            "hdr": "hdr",
            "meta": "meta",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
            "tile_stride": "tile_stride",
            "tl": "tl",
        },
        "route": {
            "compact": False,
            "kind": "persistent",
            "queue": False,
            "tagged": True,
        },
        "workspace_rule": {
            "dbg_words": 64,
            "kind": "persistent",
            "tl_words": 8,
        },
    },
    "small_k1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_d64ff2ca341acdc90cb8",
        "kernel_symbol": "kernel_cake_vsa_sm90_d64ff2ca341acdc90cb8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_d64ff2ca341acdc90cb8",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 0,
            "kind": "small",
            "kmax": 1,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k3": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_bfc9ddf76f7d5ce6a43b",
        "kernel_symbol": "kernel_cake_vsa_sm90_bfc9ddf76f7d5ce6a43b",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 116736,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_bfc9ddf76f7d5ce6a43b",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 0,
            "kind": "small",
            "kmax": 3,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k4": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_621c69308425d51cff89",
        "kernel_symbol": "kernel_cake_vsa_sm90_621c69308425d51cff89",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 149504,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_621c69308425d51cff89",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 0,
            "kind": "small",
            "kmax": 4,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k6": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_d669f7b90790e482a448",
        "kernel_symbol": "kernel_cake_vsa_sm90_d669f7b90790e482a448",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 215040,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_d669f7b90790e482a448",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 0,
            "kind": "small",
            "kmax": 6,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k4s": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "Wo"],
            ["buffer", "Ws"],
            ["buffer", "Wc"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_ebff37909a2679f79a63",
        "kernel_symbol": "kernel_cake_vsa_sm90_ebff37909a2679f79a63",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 149632,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_ebff37909a2679f79a63",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "Wc": "counters",
            "Wo": "partial_o",
            "Ws": "partial_stats",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 0,
            "kind": "small",
            "kmax": 4,
            "split": True,
        },
        "workspace_rule": {
            "counter_words": "num_tiles",
            "item_floats": 8192,
            "kind": "split",
            "stats_floats": 128,
        },
    },
    "small_k6s": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "Wo"],
            ["buffer", "Ws"],
            ["buffer", "Wc"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_861f189f5d0780fb40c2",
        "kernel_symbol": "kernel_cake_vsa_sm90_861f189f5d0780fb40c2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 215168,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_861f189f5d0780fb40c2",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "Wc": "counters",
            "Wo": "partial_o",
            "Ws": "partial_stats",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 0,
            "kind": "small",
            "kmax": 6,
            "split": True,
        },
        "workspace_rule": {
            "counter_words": "num_tiles",
            "item_floats": 8192,
            "kind": "split",
            "stats_floats": 128,
        },
    },
    "small_k2c4": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_b9403a3f1b1266dda4c1",
        "kernel_symbol": "kernel_cake_vsa_sm90_b9403a3f1b1266dda4c1",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 111616,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_b9403a3f1b1266dda4c1",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 4,
            "kind": "small",
            "kmax": 2,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k3c3": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_f18772a7614ee2d08a75",
        "kernel_symbol": "kernel_cake_vsa_sm90_f18772a7614ee2d08a75",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (3, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 151552,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_f18772a7614ee2d08a75",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 3,
            "kind": "small",
            "kmax": 3,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k3c6": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_86b6482d182122462094",
        "kernel_symbol": "kernel_cake_vsa_sm90_86b6482d182122462094",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (6, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 162816,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_86b6482d182122462094",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 6,
            "kind": "small",
            "kmax": 3,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k4c2": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_6ef70e9d44dd4f5aaefb",
        "kernel_symbol": "kernel_cake_vsa_sm90_6ef70e9d44dd4f5aaefb",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 166912,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_6ef70e9d44dd4f5aaefb",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 2,
            "kind": "small",
            "kmax": 4,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k4c4": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_3838f26cf1659246dda5",
        "kernel_symbol": "kernel_cake_vsa_sm90_3838f26cf1659246dda5",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 177152,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_3838f26cf1659246dda5",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 4,
            "kind": "small",
            "kmax": 4,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k6c2": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_e06615a69821724bb9da",
        "kernel_symbol": "kernel_cake_vsa_sm90_e06615a69821724bb9da",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232448,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_e06615a69821724bb9da",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 2,
            "kind": "small",
            "kmax": 6,
            "split": False,
        },
        "workspace_rule": None,
    },
    "small_k6sc3": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "Wo"],
            ["buffer", "Ws"],
            ["buffer", "Wc"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_599ef9958f8b6c81724d",
        "kernel_symbol": "kernel_cake_vsa_sm90_599ef9958f8b6c81724d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (3, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 215168,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_599ef9958f8b6c81724d",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "Wc": "counters",
            "Wo": "partial_o",
            "Ws": "partial_stats",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 3,
            "kind": "small",
            "kmax": 6,
            "split": True,
        },
        "workspace_rule": {
            "counter_words": "num_tiles",
            "item_floats": 8192,
            "kind": "split",
            "stats_floats": 128,
        },
    },
    "small_k6sc4": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "Wo"],
            ["buffer", "Ws"],
            ["buffer", "Wc"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_13c8e404c40cc63b7ba6",
        "kernel_symbol": "kernel_cake_vsa_sm90_13c8e404c40cc63b7ba6",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 215168,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_13c8e404c40cc63b7ba6",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "Wc": "counters",
            "Wo": "partial_o",
            "Ws": "partial_stats",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 4,
            "kind": "small",
            "kmax": 6,
            "split": True,
        },
        "workspace_rule": {
            "counter_words": "num_tiles",
            "item_floats": 8192,
            "kind": "split",
            "stats_floats": 128,
        },
    },
    "small_k6sc8": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "Vt"],
            ["buffer", "O"],
            ["param_array", "plan"],
            ["parameter", "seqlen_q"],
            ["parameter", "seqlen_k"],
            ["parameter", "scale_log2"],
            ["buffer", "Wo"],
            ["buffer", "Ws"],
            ["buffer", "Wc"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "count": "num_items",
            "kind": "plan_count",
        },
        "kernel": "cake_vsa_sm90_148864cc74c1ab7d0e21",
        "kernel_symbol": "kernel_cake_vsa_sm90_148864cc74c1ab7d0e21",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (8, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 215168,
            "use_pdl": False,
        },
        "library": "cake_vsa_sm90_148864cc74c1ab7d0e21",
        "logical": {
            "K": "k",
            "O": "out",
            "Q": "q",
            "Vt": "v",
            "Wc": "counters",
            "Wo": "partial_o",
            "Ws": "partial_stats",
            "plan": "plan",
            "scale_log2": "scale_log2",
            "seqlen_k": "seqlen_k",
            "seqlen_q": "seqlen_q",
        },
        "route": {
            "cluster": 8,
            "kind": "small",
            "kmax": 6,
            "split": True,
        },
        "workspace_rule": {
            "counter_words": "num_tiles",
            "item_floats": 8192,
            "kind": "split",
            "stats_floats": 128,
        },
    },
}

# One verified build: H100 / H200 (``sm_90a``).  The architecture is compiled
# with its exact ``-gencode`` flag (``build.json`` ``arch_flags``); there is no
# family fallback and no other Hopper variant.
ARCHES = ("sm_90a",)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
}
# The stages in registry order; the two persistent stages and the fifteen
# shipped small-selection variants.
STAGES: tuple[str, ...] = tuple(MODULES)
PERSISTENT_STAGES = (
    "attention",
    "attention_queue",
    "attention_compact",
    "attention_tagged",
)
SMALL_STAGES: tuple[str, ...] = tuple(
    name for name in MODULES if MODULES[name]["route"]["kind"] == "small"
)
# Logical argument names the plan object binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).  The
# by-value parameter arrays (``hdr`` of the persistent kernel, ``plan`` of the
# small kernel) are CPU int16 tensors copied into the kernel parameter space by
# the generated binding; everything else is a device buffer or a scalar.
LOGICAL_TENSORS = (
    "q",
    "k",
    "v",
    "out",
    "meta",
    "dbg",
    "tl",
    "partial_o",
    "partial_stats",
    "counters",
)
LOGICAL_PARAM_ARRAYS = ("hdr", "plan")
LOGICAL_SCALARS = (
    "tile_stride",
    "seqlen_q",
    "seqlen_k",
    "scale_log2",
)
GRID_NAMES = ("grid_x", "grid_y", "grid_z")


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Exported architecture of a compute capability, or ``None`` when no build exists for it."""

    return ARCH_BY_CAPABILITY.get(
        (int(device_capability[0]), int(device_capability[1]))
    )


def stage_record(name: str) -> dict[str, Any]:
    """The module record of stage ``name``; an unshipped stage (``small_k1s``, ``small_k3s``) is refused."""

    record = MODULES.get(name)
    if record is None:
        raise ValueError(
            f"unknown Cake SM90 VSA stage {name!r}; the shipped stages are {list(MODULES)}"
        )
    return record


def persistent_stage(queue: bool, compact: bool = False, tagged: bool = False) -> str:
    """The persistent kernel's stage: the queue-scheduled twin when ``queue``, the compact
    static program when ``compact`` (uniform pair plans of short tiles with many tiles per
    CTA; see ``cake_planner.COMPACT_MAX_TILE_POS``), the tagged static program when
    ``tagged`` (single-reader K / V^T tags; see ``cake_planner.kv_tags_pay``)."""

    if queue:
        return "attention_queue"
    return (
        "attention_compact"
        if compact
        else "attention_tagged"
        if tagged
        else "attention"
    )


def small_stage(kmax: int, split: bool = False, cluster: int = 0) -> str:
    """The small-selection stage of a ``plan_small`` plan: ``small_k<kmax>[s][c<cluster>]`` (``s`` and ``c``
    together name a split-cluster stage)."""

    name = f"small_k{int(kmax)}"
    if split:
        name += "s"
    if cluster:
        name += f"c{int(cluster)}"
    stage_record(name)
    return name


def route_of(name: str) -> dict[str, Any]:
    """The route a stage serves (``{"kind": "persistent", "queue"}`` or ``{"kind": "small", "kmax", "split", "cluster"}``)."""

    return dict(stage_record(name)["route"])


def workspace_rule(name: str) -> dict[str, Any] | None:
    """The plan-owned device workspace a stage needs, or ``None`` (unsplit and DSM-cluster small stages)."""

    rule = stage_record(name)["workspace_rule"]
    return None if rule is None else dict(rule)


def launch_grid(name: str, counts: Mapping[str, int]) -> tuple[int, int, int]:
    """The grid of one launch of stage ``name`` from the plan's counts (``num_ctas`` / ``num_items``)."""

    rule = stage_record(name)["grid_rule"]
    if rule["kind"] != "plan_count":
        raise ValueError(
            f"stage {name!r} has grid rule {rule['kind']!r}; this loader knows 'plan_count'"
        )
    count = int(counts[rule["count"]])
    if count < 1:
        raise ValueError(f"stage {name!r}: the plan has no {rule['count']} to launch")
    return (count, 1, 1)


def argument_names(name: str) -> tuple[str, ...]:
    """The logical names of stage ``name``'s ffi arguments in ``arg_plan`` order (``grid_x`` ... last)."""

    record = stage_record(name)
    logical = record["logical"]
    names = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            names.append(key)
            continue
        if key not in logical:
            raise ValueError(
                f"stage {name!r}: kernel parameter {key!r} has no logical name"
            )
        names.append(logical[key])
    return tuple(names)


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake SM90 VSA target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = stage_record(name)
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake SM90 VSA stage {name} is not delivered for {arch}; its architectures: {record['arches']}"
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
    name: str, values: Mapping[str, Any], *, grid: Sequence[int], device_index: int
) -> None:
    """Launch one exported stage with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_PARAM_ARRAYS`` / ``LOGICAL_SCALARS`` to tensors and Python
    scalars; the module record maps its kernel parameter names onto that
    vocabulary.  ``grid`` is the launch grid (``launch_grid``) and the kernel
    runs on the current torch stream.
    """

    record = stage_record(name)
    bound = dict(values)
    bound.update(zip(GRID_NAMES, (int(g) for g in grid), strict=True))
    args = []
    for logical_name in argument_names(name):
        if logical_name not in bound:
            raise ValueError(f"stage {name} needs {logical_name!r}")
        args.append(bound[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GRID_NAMES",
    "LOGICAL_PARAM_ARRAYS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PERSISTENT_STAGES",
    "SMALL_STAGES",
    "STAGES",
    "arch_for_capability",
    "argument_names",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "persistent_stage",
    "route_of",
    "run_module",
    "small_stage",
    "specs_for_arch",
    "stage_record",
    "workspace_rule",
]
