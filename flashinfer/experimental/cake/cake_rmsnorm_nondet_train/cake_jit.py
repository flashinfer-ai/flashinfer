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

Host plan and loader glue of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES``, ``ROUTES`` and ``HIDDEN_SIZES`` are filled by
the exporter from the complete verified program bundle and carry the host plan
only: the kernel's ``library`` stem, the architectures it is delivered for, its
launch / selection / chunking rules and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is the ordered list of module records one public call may launch.
``fwd`` / ``fwd_residual`` launch one forward program of the deterministic
package (the forward has no cross-row reduction).  ``bwd`` / ``bwd_residual``
select one of two backward forms per token-count window.  The single kernel
(workspace rule ``claim_sum_f64``) is one launch: its persistent CTAs claim row
chunks from a counter in the workspace, write ``dx`` and add their per-column
``dw`` sums into the workspace's FP64 sums with bulk reductions in the order
they finish; the CTA that finishes last stores ``dw``, the FP32 rounding of the
sums, so ``dw`` is not guaranteed bitwise reproducible.  The pair-row single
kernel (workspace rule ``claim_part_f32``) is one launch too: its CTAs keep their
``dw`` sums as FP32 pairs, write them as one row of FP32 pairs each into
scratch in the workspace, and once every row is in, helper CTAs fold the rows
and store ``dw`` (no FP64 sums; the row order follows scheduling, so ``dw`` is
not guaranteed bitwise reproducible either).  The deterministic
package's pair (``chunk_partials``; its programs renamed into this package)
is two launches: the main kernel writes one FP32 partial row of ``dw`` per
chunk, then the dw reduce kernel (role ``bwd_reduce``) folds the rows into
``dw`` in chunk order.  No launch is cooperative, so a backward never waits
for the whole device.  Token counts are runtime scalars, so a changed number
of rows never recompiles anything.

A module record is one (kernel, launch policy) pair.  A kernel launched under
several policies -- a different persistent grid or row range per stage -- has
one record per policy (``<kernel>__<stage>`` after the first), all naming the
kernel's ``library``, so it is compiled once.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Sequence

import tvm_ffi

from . import cake_build

# The three kernel-parameter plans (``arg_plan``: kernel parameter order as (kind, name);
# ``logical``: kernel parameter name -> logical argument name of the public API).
_CLASSES: dict[str, dict[str, Any]] = {
    "fwd": {
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "u"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["buffer", "h_new"],
            ["buffer", "r"],
            ["parameter", "T"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "w": "w",
            "x": "x",
            "y": "y",
        },
    },
    "bwd_main": {
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "r"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["buffer", "counters"],
            ["buffer", "g_h"],
            ["parameter", "T"],
            ["parameter", "n_chunks"],
            ["parameter", "rows_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "T": "rows",
            "counters": "counters",
            "dw": "dw",
            "dx": "dx",
            "g": "g",
            "g_h": "g_h",
            "n_chunks": "n_chunks",
            "partial": "partial",
            "r": "r",
            "rows_per_chunk": "slots",
            "w": "w",
            "x": "x",
        },
    },
    "bwd_reduce": {
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "n_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "dw": "dw",
            "n_chunks": "n_chunks",
            "partial": "partial",
        },
    },
}

# Filled mechanically from the complete verified program bundle.
MODULES: dict[str, dict[str, Any]] = {
    "cake_rmsnorm_train_03bdf5667404141e1eaf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_03bdf5667404141e1eaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03bdf5667404141e1eaf",
        "role": "fwd_mid_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_03bdf5667404141e1eaf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_e5a3aa2548fc41ee0485": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
        "role": "fwd_mid_w32_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0ad4d53c4a0df501a99d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 32,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "role": "fwd_large",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_0ad4d53c4a0df501a99d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_595f506004038a65cc63": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 32,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_595f506004038a65cc63",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_595f506004038a65cc63",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0ad4d53c4a0df501a99d__wide": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 64,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "role": "fwd_wide",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 12288 and 16384 rows: 40210950, the FP32-weight twin of bde56bd5.
    "cake_rmsnorm_train_40210950f2e7ee48bc3e__ov107_12288_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ea88523077fc340e55f": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1ea88523077fc340e55f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_21d3416baa877808371e": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ea88523077fc340e55f__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1ea88523077fc340e55f__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_21d3416baa877808371e__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_288d7b1d51c340236cb7": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_288d7b1d51c340236cb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_288d7b1d51c340236cb7",
        "role": "fwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1024 and 6144 rows: 3d619cd1, the FP32-weight twin of cf1d0554.
    "cake_rmsnorm_train_3d619cd1e3ae638cce3d__ov100_1024_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, above 6144 rows: 5f89e5fd, the FP32-weight twin of 288d7b1d.
    "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov100_6144_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1024 and 6144 rows: 2763cd5e, the FP32-weight twin of bad4db91.
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ov103_1024_6144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 6144 rows: 5f89e5fd, the FP32-weight twin of 288d7b1d.
    "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov103_6144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2a42e5c02b240f8709b1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_2a42e5c02b240f8709b1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2a42e5c02b240f8709b1",
        "role": "fwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_2a42e5c02b240f8709b1 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_79fecc7016e44e940733": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_79fecc7016e44e940733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_79fecc7016e44e940733",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2fdbab2af48db42cd2a9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_2fdbab2af48db42cd2a9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_29d0ae48de6b86aa4a41": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "role": "fwd_main",
        "select_rule": {
            "above": 1536,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_2fdbab2af48db42cd2a9__low (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_29d0ae48de6b86aa4a41__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_308da18230017dadc02d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_308da18230017dadc02d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_308da18230017dadc02d",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_308da18230017dadc02d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_8e3407e059cfc005f421": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_8e3407e059cfc005f421",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8e3407e059cfc005f421",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3430e5d9c053309b0323": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "role": "fwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3430e5d9c053309b0323 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_3c1e1f571f15d42e8d57": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3430e5d9c053309b0323__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "role": "fwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3430e5d9c053309b0323__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_3c1e1f571f15d42e8d57__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3d20ab4a432bddf7349a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3d20ab4a432bddf7349a (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_7da36ff557bb2bac8d89": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3d20ab4a432bddf7349a__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3d20ab4a432bddf7349a__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_7da36ff557bb2bac8d89__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4cdc4c495b3f59ff5069": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 8,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
        "role": "fwd_large",
        "select_rule": {
            "above": 2048,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_4cdc4c495b3f59ff5069 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_92cf7b87f1ebffb1b12b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 8,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_92cf7b87f1ebffb1b12b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_92cf7b87f1ebffb1b12b",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_60f47c5d00570ecf7740": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_60f47c5d00570ecf7740",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_60f47c5d00570ecf7740",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_60f47c5d00570ecf7740 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_61accbcc02891615f286": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_61accbcc02891615f286",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_61accbcc02891615f286",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_682675bda578f5b7a312": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_682675bda578f5b7a312",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_682675bda578f5b7a312",
        "role": "fwd_large",
        "select_rule": {
            "above": 12288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_682675bda578f5b7a312 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a30e5a3b69b7c183992f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_a30e5a3b69b7c183992f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a30e5a3b69b7c183992f",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 12288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bde56bd5a68c493f07a2": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        "role": "fwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bde56bd5a68c493f07a2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_40210950f2e7ee48bc3e": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e7f4221a7c26b9dfdd23": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_e7f4221a7c26b9dfdd23 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_fdfa500c83d95df0f5c1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fdfa500c83d95df0f5c1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fdfa500c83d95df0f5c1",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7673978dc698790054d2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_7673978dc698790054d2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_e2eabcb3b0915d916a13": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7673978dc698790054d2__quad": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_7673978dc698790054d2__quad (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_e2eabcb3b0915d916a13__quad": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_780b9833fd0a236ef0e4": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "role": "fwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_780b9833fd0a236ef0e4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a4d3f82b3e776f4fc49b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_780b9833fd0a236ef0e4__cut": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "role": "fwd_cut",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_780b9833fd0a236ef0e4__cut (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a4d3f82b3e776f4fc49b__cut": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "role": "fwd_cut_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7de1e51089d432dbc26b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7de1e51089d432dbc26b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7de1e51089d432dbc26b",
        "role": "fwd_mid",
        "select_rule": {
            "above": 3584,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_7de1e51089d432dbc26b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_1974c5c86cee63f33370": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1974c5c86cee63f33370",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1974c5c86cee63f33370",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8392d1a008d58a006bba": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_8392d1a008d58a006bba (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_4371270e4db4b985998f": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_4371270e4db4b985998f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4371270e4db4b985998f",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8392d1a008d58a006bba__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_8392d1a008d58a006bba__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_4371270e4db4b985998f__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_4371270e4db4b985998f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4371270e4db4b985998f",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8757f6da86abf54363d5": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8757f6da86abf54363d5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8757f6da86abf54363d5",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_8757f6da86abf54363d5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_6769e393999bc2ba1926": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_6769e393999bc2ba1926",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6769e393999bc2ba1926",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_a4da85136fba4588239f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a4da85136fba4588239f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a4da85136fba4588239f",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_a4da85136fba4588239f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a72c6ece1900f748eaa2": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a72c6ece1900f748eaa2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a72c6ece1900f748eaa2",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187__band": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_band",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187__band (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a__band": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_band_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187__small (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bad4db9153cc903b7ace": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bad4db9153cc903b7ace (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bad4db9153cc903b7ace__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bad4db9153cc903b7ace__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bad4db9153cc903b7ace__ps5": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 5,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bad4db9153cc903b7ace__ps5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ps5": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 5,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bf933e1660cab8e5787b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4096 and 20480 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_4096_20480": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bf933e1660cab8e5787b__wide": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "role": "fwd_wide",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 20480 and 49152 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_20480_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5a8099e177e463daab9": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c5a8099e177e463daab9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5a8099e177e463daab9",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_c5a8099e177e463daab9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_01f0ebf3db9fa2c0884e": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_01f0ebf3db9fa2c0884e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_01f0ebf3db9fa2c0884e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_cf1d05546951bdbd9f5d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_cf1d05546951bdbd9f5d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_3d619cd1e3ae638cce3d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_d447b7f6398b60fe5daa__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 1024 and 4096 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 12,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_wide",
        "select_rule": {
            "above": 3072,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_bdc524a233d2bc6e9571__wide103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 12,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 3072,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e646bbdbd852884953da": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e646bbdbd852884953da",
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_e646bbdbd852884953da (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_6325c474ad9955a319e9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "role": "fwd_main_w32_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e646bbdbd852884953da__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e646bbdbd852884953da",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_e646bbdbd852884953da__small (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_6325c474ad9955a319e9__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd913ecd254a6bba4b18 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f0db3068d0d05442a59d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 49152 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_49152_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 640,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd913ecd254a6bba4b18__low (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f0db3068d0d05442a59d__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 640,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18__low107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd913ecd254a6bba4b18__low107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f0db3068d0d05442a59d__low107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0489c784c13bb25499f6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_0489c784c13bb25499f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0489c784c13bb25499f6",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_0489c784c13bb25499f6 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f2e721db4c30900f12fe": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_f2e721db4c30900f12fe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f2e721db4c30900f12fe",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1b284baec9663d6de1b9": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1b284baec9663d6de1b9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_77456435c9fbd1d18fcf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_86d51eff64fc9fe57cf0": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "role": "fwd_rows",
        "select_rule": {
            "above": 40960,
            "at_most": 81920,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_86d51eff64fc9fe57cf0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_42d82d26e97c36651dae": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 40960,
            "at_most": 81920,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1388ab779a35c4e41efb": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_1388ab779a35c4e41efb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1388ab779a35c4e41efb",
        "role": "fwd_rows",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1388ab779a35c4e41efb (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_b0299f6e0881fb35f27b": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b0299f6e0881fb35f27b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b0299f6e0881fb35f27b",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_466331ff3cd2d6c62738": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_466331ff3cd2d6c62738 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_32bebe5a7782311d60a0": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_ed9483d8c356affee55d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_ed9483d8c356affee55d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ed9483d8c356affee55d",
        "role": "fwd_rows",
        "select_rule": {
            "above": 196608,
            "at_most": 327680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_ed9483d8c356affee55d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_d0a656f053396a58c49a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_d0a656f053396a58c49a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d0a656f053396a58c49a",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 196608,
            "at_most": 327680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 327680,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_670bfb3eed0ddf6ba461": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 327680,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bbb05314930d192dfe7b": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bbb05314930d192dfe7b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a8fffd93682c5e382dc5": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1b284baec9663d6de1b9__20480_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 20480,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1b284baec9663d6de1b9__20480_40960 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_77456435c9fbd1d18fcf__20480_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_466331ff3cd2d6c62738__81920_114688": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_466331ff3cd2d6c62738__81920_114688 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_32bebe5a7782311d60a0__81920_114688": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_86d51eff64fc9fe57cf0__114688_196608": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "role": "fwd_rows",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_86d51eff64fc9fe57cf0__114688_196608 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_42d82d26e97c36651dae__114688_196608": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_37f663183cbff23c2893": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_37f663183cbff23c2893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_37f663183cbff23c2893",
        "role": "fwd_rows",
        "select_rule": {
            "above": 327680,
            "at_most": 368640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_37f663183cbff23c2893 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_18e3c0a5204006b3a56f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 327680,
            "at_most": 368640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5__368640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 368640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5__368640_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_670bfb3eed0ddf6ba461__368640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 368640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5ca409f44e169148301": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_c5ca409f44e169148301",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5ca409f44e169148301",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_c5ca409f44e169148301 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_8f2756937eaac6b02502": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_8f2756937eaac6b02502",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8f2756937eaac6b02502",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bbb05314930d192dfe7b__6144_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "role": "fwd_rows",
        "select_rule": {
            "above": 6144,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bbb05314930d192dfe7b__6144_32768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a8fffd93682c5e382dc5__6144_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_466331ff3cd2d6c62738__32768_98304": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": 98304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_466331ff3cd2d6c62738__32768_98304 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_32bebe5a7782311d60a0__32768_98304": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 98304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_37f663183cbff23c2893__98304_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_37f663183cbff23c2893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_37f663183cbff23c2893",
        "role": "fwd_rows",
        "select_rule": {
            "above": 98304,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_37f663183cbff23c2893__98304_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_18e3c0a5204006b3a56f__98304_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 98304,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b5e173f5ecc736d91109__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "role": "fwd_mid",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 640 and 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_640_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 640 and 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_640_6144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 0 and 640 rows: ba28d2c0, the FP32-weight twin of fd1097c4.
    "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__ov100_0_640": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 640 rows: 1defffdf, the FP32-weight twin of 5eef28cd.
    "cake_rmsnorm_train_1defffdf2712cae36784__ov103_0_640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_1defffdf2712cae36784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1defffdf2712cae36784",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 6144 and 16384 rows: b49de9b4, the FP32-weight twin of 044d4a2a.
    "cake_rmsnorm_train_b49de9b47372fa4e1051__ov100_6144_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, above 16384 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_16384_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_6144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd1097c45c1e328871fd__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_fd1097c45c1e328871fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd1097c45c1e328871fd",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd1097c45c1e328871fd__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5__low107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_5eef28cd35498e9abfa5__low107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_1defffdf2712cae36784__low107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_1defffdf2712cae36784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1defffdf2712cae36784",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b5e173f5ecc736d91109__mid107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1152,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b5e173f5ecc736d91109__mid107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__mid107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1152,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_044d4a2aa0553c16db06__large107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "role": "fwd_large",
        "select_rule": {
            "above": 6144,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_044d4a2aa0553c16db06__large107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_b49de9b47372fa4e1051__large107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_044d4a2aa0553c16db06__wide107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "role": "fwd_wide",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_044d4a2aa0553c16db06__wide107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_b49de9b47372fa4e1051__wide107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 49152 rows: b49de9b4, the FP32-weight twin of 044d4a2a.
    "cake_rmsnorm_train_b49de9b47372fa4e1051__ov107_49152_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 64,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 bf16 weight: (64, 20480] backward time 0.804-1.231x the det pair's at 4 to 16384 rows over two to ten timing sessions per row count, above 1 only in one session at 5112 and 5124 rows (B200)
    "cake_rmsnorm_train_195c4317b51bb4ea1137": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H256 f32 weight: (64, 20480] backward time 0.762-1.052x the det pair's at 4 to 16384 rows over two to six timing sessions per row count, above 1 only in one session at 5112, 5120 and 5124 rows (B200)
    "cake_rmsnorm_train_8de0131445eed6eeabb7": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H512 bf16 weight: (1280, 4096] FI bench ratio < 1 against the det pair in both sessions (B200); above T 4096 the deterministic package's det pairs run for dw accuracy: this kernel's 16-step FP32 dw windows put its worst dw error at up to 1.91x the deterministic package's median FP64 error over 5 repeats at T 14928 to 65536 on B200 (rows of stride 576)
    "cake_rmsnorm_train_d0bde38aa443d42a5f04": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_d0bde38aa443d42a5f04",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d0bde38aa443d42a5f04",
        "role": "bwd_main",
        "select_rule": {
            "above": 1280,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
        },
    },
    # sm_100a H512 bf16 weight: (4096, 32768] the deterministic package's det pair of these rows (main kernel ad1a5a3c and fixed-order dw reduce 7097069b with that package's host plan), renamed into this package, for dw accuracy: the single kernel d0bde38a it replaces sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.15x to 1.91x the deterministic package's median FP64 error over 5 repeats at T 14928 to 32768 on B200 (rows of stride 576; concentrated, saturated_hi and saturated_lo inputs); the det pair took 16.22, 16.38 and 16.54 us against the single's 15.14, 15.94 and 15.97 us at T 14928, 16172 and 16384 (B200, CUDA-graph span, 2 sessions)
    "cake_rmsnorm_train_ad1a5a3c5f88e3d10036__4096_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_ad1a5a3c5f88e3d10036",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ad1a5a3c5f88e3d10036",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H512 bf16 weight: (32768, inf] the deterministic package's det pair of these rows (main kernel 8877322a and fixed-order dw reduce 122721bd with that package's host plan), renamed into this package, for dw accuracy: the single kernel d0bde38a it replaces sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.90x the deterministic package's median FP64 error over 5 repeats at T 65536 on B200 (rows of stride 576, concentrated inputs)
    "cake_rmsnorm_train_8877322a3002e3a4fefd__32768_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8877322a3002e3a4fefd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8877322a3002e3a4fefd",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H512 f32 weight: (960, inf] above T 960 for dw accuracy (over inputs 0-59 of each family the det pairs' worst dw error exceeds the FP32 eager reference's at T 1024 (0.0846 against 0.0703) and 1152 (0.0912 against 0.0832) on B200)
    "cake_rmsnorm_train_fd08c6f436eb6071fa22": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 960,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
        },
    },
    # sm_100a H1536 bf16 weight: (78784, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in registers, evict_last loads, 3 stages): it replaces the twosum single b42efd6b for speed up to T 131072; worst dw element 0.7634 of the budget at the sampled T from 78785 to 131072 on B200 (3 x 120 inputs per T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single b42efd6b keeps the window; backward time 0.88 times the replaced single's (B200)
    "cake_rmsnorm_train_66238d219de588fca483": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_66238d219de588fca483",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_66238d219de588fca483",
        "role": "bwd_main",
        "select_rule": {
            "above": 78784,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H1536 bf16 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy: the single kernel it replaces exceeds the per-element budget at T 118528 (1.029) and at larger T (2.402 at T 1048576), worst of 120 inputs of each of three families, on B200; its worst dw element stays within 0.0006 of the per-element budget at 790 sampled T from 57408 to 1048576 (every 64th T to 98304, every 256th to 131072 and 21 T to 1048576; worst of 120 inputs of each of three families in both rstd forms) on B200, and its dx is bitwise the deterministic=True backward's; backward time 1.09x to 1.11x the replaced kernel's at T 86272, 98304, 131072 and 262144 (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_b42efd6bf06148d75151__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_b42efd6bf06148d75151",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b42efd6bf06148d75151",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H1536 f32 weight: (32768, 75776] above T 32768 for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 33280 on B200, 0.6840 against 0.6219 of the per-element budget); backward time 0.941-0.965x the det pair's at T 32769, 33280, 34816 and 36864 in two timing sessions, FI bench ratio < 1 against the det pair in both sessions above T 36864 (B200); up to T 75776 its worst dw element stays within 0.793 of the per-element budget at every sampled T (every 256th row from 32768 to 65536 and every 64th from 65536 to 75776, worst of 120 inputs of each of three families in both rstd forms) on B200; it exceeds the budget at T 108736 (1.060); the twosum form of this kernel runs above T 75776; the twosum form takes 1.05x to 1.12x this kernel's backward time from T 32769 to 75776 (B200)
    "cake_rmsnorm_train_4b6981c10fd40a9d8aec__32768_75776": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 75776,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H1536 f32 weight: (75776, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in registers, evict_last loads, 4 stages): it replaces the twosum single e2b99fee for speed up to T 131072; worst dw element 0.7705 of the budget and at or below the FP32 eager reference's at the sampled T from 75777 to 131072 on B200 (3 x 120 inputs per T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single e2b99fee keeps the window; backward time 0.89 to 0.90 times the replaced single's (B200)
    "cake_rmsnorm_train_5e78d78f2d8203948198": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_5e78d78f2d8203948198",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5e78d78f2d8203948198",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 75776,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H1536 f32 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces exceeds the per-element budget at T 108736 (1.060, worst of 120 inputs of each of three families) on B200; its worst dw element stays within 0.0006 of the per-element budget at every sampled T from 75777 to 131072 and at T 1048576 (72 T, worst of 120 inputs of each of three families in both rstd forms) on B200; backward time 1.10x to 1.11x the replaced kernel's at T 75777, 98304 and 131072 (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_e2b99fee32764cfd1295__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_e2b99fee32764cfd1295",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e2b99fee32764cfd1295",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H1536 f32 weight: (1024, 4096] the single kernel of its largest token counts for dw accuracy: the chunk-partials pair of (1024, 4096] exceeded the FP32 eager reference's dw error at T 3136 on B200 (the deterministic package runs the same pair there); its worst dw error stays within 0.1063 of the per-element budget and at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges (49 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.02x to 1.23x the programs' it replaces at T 1025, 1152, 1280, 1408, 1536, 1792, 2048, 3136 and 4096 in two sessions (B200)
    "cake_rmsnorm_train_4b6981c10fd40a9d8aec__1024_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H5120 f32 weight: (1792, 6144] the single kernel whose twosum form runs its top window for dw accuracy: the chunk-partials pair of (256, 6144] exceeded the FP32 eager reference's dw error at T 2752, 3840, 4224, 4352 and 5504 on B200 (the deterministic package runs the same pair there); its worst dw error stays within 0.2149 of the per-element budget and at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges (68 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.14x to 1.21x the programs' it replaces at T 2048, 2304, 2560, 2752, 3072, 4352 and 6144 in two sessions (B200)
    "cake_rmsnorm_train_de8f6091077dc7ac5947__1792_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_de8f6091077dc7ac5947",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_de8f6091077dc7ac5947",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1792,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_100a H2048 f32 weight: (336, 1024] the column-task kernel of its lowest window for dw accuracy: the chunk-partials pair of (336, 3072] exceeded the FP32 eager reference's dw error at T 1728 and 2773 on B200 (the deterministic package runs the same pair there); its worst dw error stays within 0.0375 of the per-element budget and at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges (12 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.86x to 1.15x the programs' it replaces at T 337, 400, 480, 560, 640, 720, 800, 880, 960 and 1024 in two sessions (B200)
    "cake_rmsnorm_train_17493054fed5409b6df9__336_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_17493054fed5409b6df9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_17493054fed5409b6df9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 336,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (42496, 65536] FI bench ratio < 1 against the det pair in both sessions (B200); up to T 65536 its worst dw element stays within 0.751 of the per-element budget at every sampled T from 5633 to 65536 (75 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 72 of them) on B200; its worst dw error was 1.07x the deterministic package's median FP64 error over 5 repeats at T 65536 on B200 (concentrated inputs); below T 42496 the deterministic package's det pair runs for dw accuracy; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_9433061da18b2db4299d__42496_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_9433061da18b2db4299d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9433061da18b2db4299d",
        "role": "bwd_main",
        "select_rule": {
            "above": 42496,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 bf16 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; help finish, pairs in registers, evict_last loads, 4 stages): it replaces the twosum single 10a41132 for speed up to T 131072; worst dw element 0.9091 of the budget at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single 10a41132 keeps the window; backward time 0.93 times the replaced single's (B200)
    "cake_rmsnorm_train_995b2229fe43c772b2c7": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_995b2229fe43c772b2c7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_995b2229fe43c772b2c7",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 bf16 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces above T 65536 exceeds the per-element budget at T 105259 (1.047, measured with FP32 weight; the BF16-weight program computes the same dw) on B200; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.07x to 1.08x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_10a41132a42d594ce6dd__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_10a41132a42d594ce6dd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_10a41132a42d594ce6dd",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 f32 weight: (3584, 65536] backward time 0.970-1.001x the det pair's at T 4096, 14928, 16384, 32768 and 65536 over four timing sessions (B200); up to T 65536 its worst dw element stays within 0.790 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 3585 to 65536 (196 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 195 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_a562656cef5bf5dda160__3584_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a562656cef5bf5dda160",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a562656cef5bf5dda160",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 f32 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; help finish, pairs in registers, evict_last loads, 4 stages): it replaces the twosum single 45cf60e3 for speed up to T 131072; worst dw element 0.8561 of the budget and at or below the FP32 eager reference's at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single 45cf60e3 keeps the window; backward time 0.92 to 0.93 times the replaced single's (B200)
    "cake_rmsnorm_train_d5a17016b7ff2c6e3ba9": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_d5a17016b7ff2c6e3ba9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d5a17016b7ff2c6e3ba9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 f32 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces above T 65536 exceeds the per-element budget at T 105259 (1.047) on B200; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.09x to 1.10x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_45cf60e3cd8603c2abeb__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_45cf60e3cd8603c2abeb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_45cf60e3cd8603c2abeb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual bf16 weight: (1408, 65536] the twosum help single kernel 1609f796 at stages=3, for speed with the dw arithmetic of 1609f796, which runs these rows for dw accuracy (its twosum form: exact per-thread dw sums; the plain form 3fbfdd75 sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.16x to 1.97x the deterministic package's median FP64 error over 5 repeats at T 14928 to 65536 on B200): on B200 (CUDA graph replay, cold L2, paired rounds) at T 1409 to 65536 it took 1.04x to 1.05x the time of the program FlashInfer e14ddfd9 runs at T <= 2048, 1.01x to 1.02x at T 3072 to 4096 and 0.97x to 0.99x from T 6144 (1609f796: 0.98x to 1.09x), with the dw error of 1609f796
    "cake_rmsnorm_train_eb332e8c97f6387027c6__1408_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_eb332e8c97f6387027c6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_eb332e8c97f6387027c6",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1408,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual bf16 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in the shared row, evict_last loads, 4 stages): it replaces the twosum single 1609f796 for speed up to T 131072; worst dw element 0.8328 of the budget at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single 1609f796 keeps the window; backward time 0.92 to 0.94 times the replaced single's (B200)
    "cake_rmsnorm_train_7c216db7f9e0c6fd9c4d": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_7c216db7f9e0c6fd9c4d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c216db7f9e0c6fd9c4d",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual bf16 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) up to T 131072, the top of the qualified range (the single kernel it replaces stays within the per-element budget at every sampled T above T 65536, worst 0.903 at T 126976, on B200, worst of 60 inputs of each of three families in both rstd forms, not yet qualifying evidence); its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.00x to 1.01x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_1609f7969c50fba7f2c7__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_1609f7969c50fba7f2c7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1609f7969c50fba7f2c7",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual f32 weight: (1408, 65536] FI bench ratio < 1 against the det pair in both sessions (B200); up to T 65536 its worst dw element stays within 0.723 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 1409 to 65536 (168 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 167 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_a640d0b799889f785ad5__1408_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a640d0b799889f785ad5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a640d0b799889f785ad5",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1408,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual f32 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in the shared row, evict_last loads, 4 stages): it replaces the twosum single d3d67437 for speed up to T 131072; worst dw element 0.8547 of the budget and at or below the FP32 eager reference's at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single d3d67437 keeps the window; backward time 0.92 to 0.93 times the replaced single's (B200)
    "cake_rmsnorm_train_0bdcd2c63a9a4c5cb9d4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_0bdcd2c63a9a4c5cb9d4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0bdcd2c63a9a4c5cb9d4",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual f32 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) up to T 131072, the top of the qualified range (the single kernel it replaces stays within 0.836 of the per-element budget at 13 sampled T above T 65536 on B200, worst of 60 inputs of each of three families in both rstd forms, not yet qualifying evidence); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.00x to 1.01x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_d3d674373d17aaf425c0__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_d3d674373d17aaf425c0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d3d674373d17aaf425c0",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H2048 bf16 weight: (12, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.32x to 1.59x the det package's speed and 1.33x to 1.63x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_ef052c0918725ffa68d4__12_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ef052c0918725ffa68d4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ef052c0918725ffa68d4",
        "role": "bwd_main",
        "select_rule": {
            "above": 12,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 bf16 weight: (64, 336] FI bench ratio < 1 against the det pair in both sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_64e6aa82bb2c6d4ea978__64_336": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 336,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (14, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.39x to 1.53x the det package's speed and 1.35x to 1.56x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_382e45788ace3ff40ba1__14_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_382e45788ace3ff40ba1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_382e45788ace3ff40ba1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 14,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (64, 336] FI bench ratio < 1 against the det pair in both sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_17493054fed5409b6df9__64_336": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_17493054fed5409b6df9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_17493054fed5409b6df9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 336,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (12, 64] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.27x to 1.49x the det package's speed and 1.09x to 1.30x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_231e01eca0459c2c0bf5__12_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_231e01eca0459c2c0bf5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_231e01eca0459c2c0bf5",
        "role": "bwd_main",
        "select_rule": {
            "above": 12,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (64, 1152] backward time 0.925-1.020x the det pair's at T 256 and 1024 over four timing sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_4822793ea0c954c11af6__64_1152": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_4822793ea0c954c11af6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4822793ea0c954c11af6",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (14, 64] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.20x to 1.40x the det package's speed and 1.13x to 1.31x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_ce1f7d216ecc0a1573b9__14_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_ce1f7d216ecc0a1573b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ce1f7d216ecc0a1573b9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 14,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (64, 1792] FI bench ratio < 1 against the det pair in both sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_8616070567821eaa3e35__64_1792": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_8616070567821eaa3e35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8616070567821eaa3e35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1792,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 bf16 weight: (4, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.18x to 1.41x the det package's speed and 1.16x to 1.42x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__4_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 bf16 weight: (64, 896] backward time 0.923-1.014x the det pair's at T 256 over eight timing sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__64_896": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (4, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.42x to 2.09x the det package's speed and 1.21x to 1.51x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_d70f2562a592151ec4f9__4_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_d70f2562a592151ec4f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d70f2562a592151ec4f9",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (64, 1152] FI bench ratio < 1 against the det pair in both sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_44508df7dab2ab688f85__64_1152": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 f32 weight: (4, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.37x to 2.15x the det package's speed and 1.22x to 1.43x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_9b37923697ee8ca52a9e__4_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_9b37923697ee8ca52a9e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9b37923697ee8ca52a9e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 f32 weight: (64, 1168] FI bench ratio < 1 against the det pair in both sessions (B200); the det pair runs above T 1168 (faster from T 1184 in every session, 1.05x at T 1408); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_b3e6531e46667901ab91__64_1168": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (2, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.31x to 2.04x the det package's speed and 1.24x to 1.51x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_7b0f5a6a8769473c322e__2_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 2,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (64, 1408] FI bench ratio < 1 against the det pair in both sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_864e84f3d874f850c091__64_1408": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 64,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual f32 weight: (4, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.32x to 2.10x the det package's speed and 1.25x to 1.46x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_ac9d69f94aabfc68aa40__4_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual f32 weight: (64, 1408] FI bench ratio < 1 against the det pair in both sessions (B200); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_71d36617003ab7742a77__64_1408": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 64,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (4608, 16384] the twosum help single kernel c67fc8b9 with evict_last loads and without r_ahead, for speed with the dw arithmetic of c67fc8b9, which runs these rows for dw accuracy (its twosum form: exact per-thread dw sums; the plain form 15e157d2 sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.10x to 2.30x the deterministic package's median FP64 error over 5 repeats at T 14928 to 16384 on B200): on B200 (CUDA graph replay, cold L2, paired rounds) at T 4609 to 16384 it took 1.00x to 1.04x the time of the program FlashInfer e14ddfd9 runs (c67fc8b9: 1.01x to 1.06x), with the dw error of c67fc8b9 (q <= 0.987 of the deterministic package's program at FlashInfer 2e989556)
    "cake_rmsnorm_train_2443b1f0183048c47b15__4608_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_2443b1f0183048c47b15",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2443b1f0183048c47b15",
        "role": "bwd_main",
        "select_rule": {
            "above": 4608,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 bf16 weight: (16384, 36864] the twosum help single kernel c67fc8b9 with evict_last loads, for speed with the dw arithmetic of c67fc8b9 (the reason of (4608, 16384]): on B200 (CUDA graph replay, cold L2, paired rounds) it took 0.99x to 1.01x the time of the program FlashInfer e14ddfd9 runs from T 20480 to 36864 (c67fc8b9: 1.01x to 1.06x over T 4609 to 36864), with the dw error of c67fc8b9 (q <= 0.987)
    "cake_rmsnorm_train_03f6488b89b5f9ae3db5__16384_36864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_03f6488b89b5f9ae3db5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03f6488b89b5f9ae3db5",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 f32 weight: (3584, 36864] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_f7573bcb748c0784f3ec": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 f32 weight: (1792, 3584] the single kernel of its largest token counts for dw accuracy: the chunk-partials pair of (1792, 3584] exceeded the FP32 eager reference's dw error at T 2569 on B200 (the deterministic package runs the same pair there); its worst dw error stays within 0.1404 of the per-element budget and at or below the FP32 eager reference's at every multiple of 64 rows from 1856 to 3584 on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.96x to 1.05x the pair's at T 2048, 2569 and 3584 in two sessions (B200)
    "cake_rmsnorm_train_f7573bcb748c0784f3ec__1792_3584": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1792,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H2048 bf16 weight: (7168, 11264] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_0fa4f255caf709d09f66": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0fa4f255caf709d09f66",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0fa4f255caf709d09f66",
        "role": "bwd_main",
        "select_rule": {
            "above": 7168,
            "at_most": 11264,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 bf16 weight: (16384, 45056] the deterministic package's det pair of these rows (main kernel ceeef043 and fixed-order dw reduce d0960be9 with that package's host plan), renamed into this package, for dw accuracy: the single kernel 0fa4f255 it replaces sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.25x the deterministic package's median FP64 error over 5 repeats at T 32768 on B200 (concentrated inputs); up to T 58880, the deterministic package's own edge, since 0fa4f255 also exceeds 1.1x that package's median dw error at T 51968 on B200 (off the gate's rows), where the det pair takes 0.973x to 0.984x its backward time
    "cake_rmsnorm_train_ceeef043ac64f37fc529__16384_58880": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ceeef043ac64f37fc529",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ceeef043ac64f37fc529",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H2048 f32 weight: (15403, 45056] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_ba7dbf6a07744268705e": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 15403,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 f32 weight: (45056, 58880] the single kernel of its window above, within 2 % of the det pair's backward time (the det pair of the window, the deterministic package's program, reaches 0.900 of the per-element budget, K form, worst of 120 inputs of each of three families on B200); this single kernel's worst dw element stays within 0.777 of the per-element budget and at or below the FP32 eager reference's at every 64th T of the window (worst of 120 inputs of each of three families in both rstd forms) on B200; backward time 0.98x to 1.01x the det pair's at T 45057, 51968 and 58880 (B200)
    "cake_rmsnorm_train_ba7dbf6a07744268705e__45056_58880": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 45056,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 f32 weight: (1024, 4096] the single kernel whose twosum form runs its top window for dw accuracy: the chunk-partials pairs of (336, 3072] and (3072, 4096] exceeded the FP32 eager reference's dw error at T 1728 and 2773 and at T 3920 on B200 (the deterministic package runs the same pairs there); its worst dw error stays within 0.1265 of the per-element budget and at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges (50 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.86x to 1.17x the programs' it replaces at T 1152, 1728, 2773, 3072 and 3920 in two sessions (B200)
    "cake_rmsnorm_train_ba7dbf6a07744268705e__1024_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_103a H5120 bf16 weight: (61952, 393216] above T 61952 for dw accuracy (both weight dtypes switch at T 61952: the FP32-weight route's compensated det pair, the same dw computation, reaches 0.888 of the per-element budget at T 62464 on GB300); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.588 of the budget at the sampled T from 61953 to 131072 on GB300 (3 x 60 inputs, kernel and torch rstd); its FP32-weight twin, the same dw computation, 0.688 at 73 sampled T from 61953 to 131072, 1.16 times the twosum single's speed in the FI bench (GB300, 4 sessions); up to T 393216, the largest sampled T at or below which its worst dw element stays within the per-element budget (0.993 of it); at T 524288 it is 1.100 of it (GB300, 3 x 120 inputs per T, 3 x 240 at T 262144 and 393216, kernel and torch rstd)
    "cake_rmsnorm_train_b523a764b029e5ca1aaf__61952_393216": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b523a764b029e5ca1aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b523a764b029e5ca1aaf",
        "role": "bwd_main",
        "select_rule": {
            "above": 61952,
            "at_most": 393216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 81920,
        },
    },
    # sm_103a H5120 bf16 weight: (393216, inf] the twosum single above T 393216 for dw accuracy (the kblock single below exceeds the per-element budget at T 524288, 1.100 of it, on GB300); twosum: exact per-thread dw sums, so the dw error does not grow with T: worst dw element 0.0006 of the budget at the sampled T from 524288 to 1048576 on GB300 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_9e4c0825df92af02074f__393216_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9e4c0825df92af02074f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e4c0825df92af02074f",
        "role": "bwd_main",
        "select_rule": {
            "above": 393216,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 81920,
        },
    },
    # sm_103a H5120 f32 weight: (61952, 196608] above T 61952 for dw accuracy (the compensated det pair, from T 5120, reaches 0.888 of the per-element budget at T 62464 on GB300); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.688 of the budget at 73 sampled T from 61953 to 131072 on GB300 (3 x 60 inputs per T, kernel and torch rstd), at most the FP32 eager reference's, 1.18 times the twosum single's speed in the FI bench (GB300, 4 sessions); up to T 196608, the largest sampled T at or below which its worst dw element stays within the per-element budget (0.700 of it); at T 262144 it is 1.004 of it (GB300, 3 x 120 inputs per T, 3 x 240 at T 262144, kernel and torch rstd)
    "cake_rmsnorm_train_90598c14b4e396f5c8b0__61952_196608": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_90598c14b4e396f5c8b0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_90598c14b4e396f5c8b0",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 61952,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 81920,
        },
    },
    # sm_103a H5120 f32 weight: (196608, inf] the twosum single above T 196608 for dw accuracy (the kblock single below exceeds the per-element budget at T 262144, 1.004 of it, on GB300); twosum: exact per-thread dw sums, so the dw error does not grow with T: worst dw element 0.0006 of the budget at the sampled T from 262144 to 1048576 on GB300 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_f165933caadda3434828__196608_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_f165933caadda3434828",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f165933caadda3434828",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 196608,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 81920,
        },
    },
    # sm_103a H6144 residual bf16 weight: (6, 31] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.08x to 1.47x the det package's speed and 1.10x to 1.49x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_7b0f5a6a8769473c322e__6_31": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 6,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (31, 32] the deterministic package's column-task program of these token counts (96 column tasks), for speed: 0.80x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 17 to 32 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_07135550fae8b07bb82c__31_32": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_07135550fae8b07bb82c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07135550fae8b07bb82c",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 31,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (32, 48] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.08x to 1.47x the det package's speed and 1.10x to 1.49x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_7b0f5a6a8769473c322e__32_48": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 32,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (48, 56] backward time 0.69x the det pair's at T 8 and 0.93x at T 56 (GB300); from T 56 to 304 the det pair runs (1.04-1.10x faster at T 57 to 64 and 280 to 304); T 1 to 48 runs the fixed-claim column-task form
    "cake_rmsnorm_train_864e84f3d874f850c091__48_56": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 48,
            "at_most": 56,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (304, 1920] backward time 0.88x the det pair's at T 312, 0.85x at T 512, 0.89x at T 1024 and 0.95x at T 1792 (GB300); level at T 1920, from where the det pair runs (1.04x faster at T 1984)
    "cake_rmsnorm_train_864e84f3d874f850c091__304_1920": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 304,
            "at_most": 1920,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (320, 1824] FI bench ratio < 1 against the det pair in both sessions (GB300; 0.95x at T 1824); the det pair runs above T 1824 (1.03x faster at T 1832 to 1856)
    "cake_rmsnorm_train_71d36617003ab7742a77__320_1824": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 320,
            "at_most": 1824,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (7168, 131072] the pair-row single kernel with its chunk table cut for 4 slots per SM (table_slots 4: more, shorter chunks than CTAs, the grid unchanged; twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on R200 with 212 SMs (CUDA graph replay, cold L2, 7 rounds) it took 41.6 to 555 us at T 7169 to 131072, 0.80x to 0.81x the time of the twosum single kernel c07ab5f1 that FlashInfer main runs there, with a dw error (relative L2 or worst element, the larger over six input families, 3 seeds x 5 repeats; dx equal to the deterministic package's) of at most 1.047x that of the deterministic package's program at FlashInfer 2e989556, except 1.133x at T 8192 (saturated_lo inputs), equal to c07ab5f1's there
    "cake_rmsnorm_train_16c5f0465183f87c5229__7168_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_16c5f0465183f87c5229",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_16c5f0465183f87c5229",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 7168,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_107a H6144 residual bf16 weight: (131072, inf] above the pair-row single's measured range (16c5f046 runs (7168, 131072]), the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces exceeds the per-element budget at T 127424 (1.014, measured with FP32 weight; the BF16-weight program computes the same dw) on R200 with 212 SMs; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 7169 to 131072 (65 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.96x to 0.98x the single kernel's at T 7169, 32768, 65536, 98304 and 131072 in two sessions (R200, 212 SMs); faster than the deterministic package's program for those token counts too (0.95x to 0.99x its time at the same T)
    "cake_rmsnorm_train_c07ab5f16934c0cc8421__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c07ab5f16934c0cc8421",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c07ab5f16934c0cc8421",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_107a H6144 residual f32 weight: (7168, 131072] the FP32-weight twin of the pair-row single kernel of the BF16-weight route (table_slots 4; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on R200 with 212 SMs (CUDA graph replay, cold L2, 7 rounds) it took 41.0 to 556 us at T 7169 to 131072, 0.78x to 0.82x the time of the twosum single kernel e655055e that FlashInfer main runs there, with a dw error (relative L2 or worst element, the larger over six input families, 3 seeds x 5 repeats; dx equal to the deterministic package's) of at most 1.025x that of the deterministic package's program at FlashInfer 2e989556
    "cake_rmsnorm_train_999d4acb1d1a4e0cbef0__7168_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_999d4acb1d1a4e0cbef0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_999d4acb1d1a4e0cbef0",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 7168,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_107a H6144 residual f32 weight: (131072, inf] above the pair-row single's measured range (999d4acb runs (7168, 131072]), the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces exceeds the per-element budget at T 127424 (1.014) on R200 with 212 SMs; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 7169 to 131072 (131 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.96x to 0.99x the single kernel's at T 14337, 32768, 65536, 98304 and 131072 in two sessions (R200, 212 SMs); faster than the deterministic package's program for those token counts too (0.95x to 0.97x its time at the same T)
    "cake_rmsnorm_train_e655055ead76f8df4d24__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e655055ead76f8df4d24",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e655055ead76f8df4d24",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_107a H6144 residual bf16 weight: (576, 3584] backward time 1.009-1.046x the det pair's at T 1024 in four timing sessions (R200, 212 SMs)
    "cake_rmsnorm_train_864e84f3d874f850c091__576_3584": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 576,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (768, 3584] backward time 1.039-1.050x the det pair's at T 1024 in four timing sessions (R200, 212 SMs)
    "cake_rmsnorm_train_71d36617003ab7742a77__768_3584": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 768,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H256 bf16 weight: (524288, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T (from 92160 to 524288 rows the deterministic package's det pair runs: its worst dw element stays within 0.69 of the per-element budget at the sampled row counts from 80896 to 524288 rows on B200, both weight dtypes, worst of 60 inputs of each of three families, and exceeds the budget above 864240 rows)
    "cake_rmsnorm_train_172a47a15d4c1a76c061": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "role": "bwd_main",
        "select_rule": {
            "above": 524288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H256 bf16 weight: (20480, 92160] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above 20480 rows; from 92160 to 524288 rows the deterministic package's det pair for those rows runs (main kernel 083ea823, dw reduce b4b63d5b: faster at every sampled row count above 98304 on B200, level with this form from 80896 to 92160 rows)
    "cake_rmsnorm_train_172a47a15d4c1a76c061__20480_92160": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "role": "bwd_main",
        "select_rule": {
            "above": 20480,
            "at_most": 92160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    "cake_rmsnorm_train_083ea823c691e0631ad3__92160_524288": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "role": "bwd_main",
        "select_rule": {
            "above": 92160,
            "at_most": 524288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b4b63d5b5028c7a60333__92160_524288": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 92160,
            "at_most": 524288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 f32 weight: (92160, 864240] the dw reduce of the det pair c4c89f0d of that window (the BF16-weight route keeps b4b63d5b__92160_524288)
    "cake_rmsnorm_train_b4b63d5b5028c7a60333__92160_864240": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 92160,
            "at_most": 864240,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 f32 weight: (864240, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T (from 92160 to 864240 rows the deterministic package's det pair runs: its worst dw element stays within 0.69 of the per-element budget at the sampled row counts from 80896 to 524288 rows on B200, worst of 60 inputs of each of three families, and within 0.52 at 589824, 786432 and 864240 rows, worst of 100 inputs of each of two families; against the FP64 reference at the FP64 rstd it reaches 1.11 of the budget at 864264 rows)
    "cake_rmsnorm_train_7c2aa4b429538da06a2a__864240_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 864240,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H256 f32 weight: (20480, 92160] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above 20480 rows; from 92160 to 524288 rows the deterministic package's det pair for those rows runs (main kernel c4c89f0d, dw reduce b4b63d5b; one cut with the BF16-weight route)
    "cake_rmsnorm_train_7c2aa4b429538da06a2a__20480_92160": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 92160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H256 f32 weight: (92160, 864240] the deterministic package's det pair of these rows (main kernel c4c89f0d and fixed-order dw reduce b4b63d5b with that package's host plan, det record (80896, 1572864]), renamed into this package and cut to this window (the BF16-weight route cuts its pair at 524288 rows); from 524288 rows for speed: on B200 its backward took 0.887x to 0.899x the time of the twosum single 7c2aa4b4 at 524304 to 864240 rows (FI bench, L2 flushed before every call, two runs), and it is the program the deterministic package at FlashInfer 2e989556 runs there
    "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__92160_864240": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 92160,
            "at_most": 864240,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H6144 residual bf16 weight: (1920, 131072] the pair-row single kernel (twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on GB300 (CUDA graph replay, cold L2) it took 25.7 to 909 us at T 1921 to 131072, 0.84x to 0.94x the time of the programs FlashInfer e14ddfd9 runs at those T, with a dw error (relative L2 or worst element, the larger over six input families) of at most 1.0x that of, except 1.2x at T 98304 (equal to the twosum single's there), the deterministic package's program at FlashInfer 2e989556
    "cake_rmsnorm_train_601fb35c90bc116731b1__1920_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_601fb35c90bc116731b1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_601fb35c90bc116731b1",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1920,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_103a H6144 residual bf16 weight: (131072, inf] above the pair-row single's measured range, the twosum form of the single kernel (its knobs plus twosum: exact per-thread dw sums, so the dw error does not grow with T) for dw accuracy: the plain form's 32-row FP32 dw sums put its worst dw error at 1.28x to 3.56x the deterministic package's median FP64 error over 5 repeats at T 14928 to 32768 on GB300 (BF16 weight, all five input families); within 3 % of the plain form's time
    "cake_rmsnorm_train_52831494c1b62db758fd": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_52831494c1b62db758fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_52831494c1b62db758fd",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 residual f32 weight: (1824, 131072] the pair-row single kernel (twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on GB300 (CUDA graph replay, cold L2) it took 24.3 to 907 us at T 1825 to 131072, 0.85x to 0.92x the time of the programs FlashInfer e14ddfd9 runs at those T, with a dw error (relative L2 or worst element, the larger over six input families) of at most 1.0x that of, except 1.2x at T 98304 (equal to the twosum single's there), the deterministic package's program at FlashInfer 2e989556; the FP32-weight twin computes the BF16-weight program's dw
    "cake_rmsnorm_train_cf1cf64436893a2a2d03__1824_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_cf1cf64436893a2a2d03",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cf1cf64436893a2a2d03",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1824,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_103a H6144 residual f32 weight: (131072, inf] above the pair-row single's measured range, the twosum form of the single kernel (its knobs plus twosum: exact per-thread dw sums, so the dw error does not grow with T) for dw accuracy: the plain form's 32-row FP32 dw sums put its worst dw error at 1.28x to 3.56x the deterministic package's median FP64 error over 5 repeats at T 14928 to 32768 on GB300 (BF16 weight, all five input families); the FP32-weight twins compute the same dw; within 3 % of the plain form's time
    "cake_rmsnorm_train_56c46a201c168050aa60": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_56c46a201c168050aa60",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56c46a201c168050aa60",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H1536 bf16 weight: (46592, 53184] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.241 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.403 of the budget at the sampled T up to 53184
    "cake_rmsnorm_train_1df502fd0e17091869b7__46592_53184": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1df502fd0e17091869b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1df502fd0e17091869b7",
        "role": "bwd_main",
        "select_rule": {
            "above": 46592,
            "at_most": 53184,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 bf16 weight: (64, 214] the deterministic package's column-task program of these token counts (32 column tasks), for speed: 0.72x to 0.82x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 65 to 214 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_53285b9598170177d3e0__64_214": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_53285b9598170177d3e0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_53285b9598170177d3e0",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (53184, 76352] the deterministic package's program as an ND pair for speed: backward 0.977-0.978 of the kblock single's time (R2-65 GB300 records); worst dw element 0.904 of the budget at the sampled T (GB300, at least 3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_cbeae168d70edd196417__53184_76352": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_cbeae168d70edd196417",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cbeae168d70edd196417",
        "role": "bwd_large",
        "select_rule": {
            "above": 53184,
            "at_most": 76352,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_882ec880717f382e43eb__53184_76352": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_882ec880717f382e43eb",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_882ec880717f382e43eb",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 53184,
            "at_most": 76352,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 bf16 weight: (76352, 131072] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.244 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.724 of the budget at the sampled T up to 131072
    "cake_rmsnorm_train_1df502fd0e17091869b7__76352_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1df502fd0e17091869b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1df502fd0e17091869b7",
        "role": "bwd_main",
        "select_rule": {
            "above": 76352,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 bf16 weight: (131072, inf] the twosum single above T 131072: the kblock single's dw accuracy is qualified to T 131072 only (its worst dw element is 1.446 of the budget at T 1048576 on GB300)
    "cake_rmsnorm_train_327bfeb4ab8fefc2731d__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_327bfeb4ab8fefc2731d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_327bfeb4ab8fefc2731d",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 f32 weight: (37888, 40448] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.256 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.416 of the budget at the sampled T up to 40448
    "cake_rmsnorm_train_9a47ed21f63c848f2a59__37888_40448": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9a47ed21f63c848f2a59",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a47ed21f63c848f2a59",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 37888,
            "at_most": 40448,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 f32 weight: (40448, 76352] the deterministic package's program as an ND pair for speed: backward 0.962-0.972 of the kblock single's time (R2-65 GB300 records); worst dw element 0.904 of the budget at the sampled T (GB300, at least 3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_ca6211f3ea2ce8c86a05__40448_76352": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_ca6211f3ea2ce8c86a05",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ca6211f3ea2ce8c86a05",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 40448,
            "at_most": 76352,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_882ec880717f382e43eb__40448_76352": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_882ec880717f382e43eb",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_882ec880717f382e43eb",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 40448,
            "at_most": 76352,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 f32 weight: (76352, 131072] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.253 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.707 of the budget at the sampled T up to 131072
    "cake_rmsnorm_train_9a47ed21f63c848f2a59__76352_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9a47ed21f63c848f2a59",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a47ed21f63c848f2a59",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 76352,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 f32 weight: (131072, inf] the twosum single above T 131072: the kblock single's dw accuracy is qualified to T 131072 only (its worst dw element is 1.363 of the budget at T 1048576 on GB300)
    "cake_rmsnorm_train_963a7ca8d03f78bbe6f9__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_963a7ca8d03f78bbe6f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_963a7ca8d03f78bbe6f9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H4096 bf16 weight: (32, 152] the deterministic package's column-task program of these token counts (128 column tasks), for speed: 0.84x to 0.89x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 33 to 152 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_ff0156d59b2b99144701__32_152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_ff0156d59b2b99144701",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ff0156d59b2b99144701",
        "role": "bwd_main",
        "select_rule": {
            "above": 32,
            "at_most": 152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 bf16 weight: (10239, 65536] the pair-row single kernel with its chunk table cut for 8 slots per SM (table_slots 8: more, shorter chunks than CTAs, the grid unchanged; 128-thread CTAs of 32 items, two per SM, 128 helper CTAs) for speed: on GB300 (CUDA graph replay, cold L2, 7 rounds) it took 125.89 us and 237.73 us at T 32768 and 65536, 0.953x and 0.938x the time of the programs FlashInfer main runs there (the det pair 35a0b0ca, the twosum single 8250eb92), with a dw error (relative L2 or worst element, the larger over six input families) of 0.975x and 0.860x that of the deterministic package's program at FlashInfer 2e989556; from T 10240 to 32768, where the det pair 35a0b0ca ran before, its backward took 0.940x to 0.959x the det pair's time on GB300 (FI bench, L2 flushed before every call, two runs at T 10240, 12288, 14928, 15288, 15566, 16114, 16164, 16384, 16385, 18432, 20480, 24576, 28672 and 32768), with a dw error of at most 0.979x the median of the deterministic package's program at FlashInfer 2e989556 (six input families, three seeds, five repeats) at the gated T; the dw gate has not sampled T 10240 yet: it passes at T 8192 and 12288 next to it, and the worst dw element stays within 0.0006 of the per-element budget at T 8192 and 16384 (0.125 and 0.173 against the FP64-rstd reference); below T 10240 the det pair stays: there a forward and backward call or a training step was slower with this kernel in both runs at T 4097, 4608 and 8192
    "cake_rmsnorm_train_06e5f7e09e42d1108365__10239_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 8,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_06e5f7e09e42d1108365",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_06e5f7e09e42d1108365",
        "role": "bwd_main",
        "select_rule": {
            "above": 10239,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 32768,
        },
    },
    # sm_103a H4096 bf16 weight: (65536, 262144] the pair-row single kernel (twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on GB300 (CUDA graph replay, cold L2) it took 196 to 952 us at T 50688 to 262144, 0.97x to 0.98x the time of the programs FlashInfer e14ddfd9 runs at those T, with a dw error (relative L2 or worst element, the larger over six input families) of at most 0.86x that of the deterministic package's program at FlashInfer 2e989556
    "cake_rmsnorm_train_3e6231b66942e510fdae__65536_262144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_3e6231b66942e510fdae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3e6231b66942e510fdae",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 32768,
        },
    },
    # sm_103a H4096 bf16 weight: (262144, inf] the twosum single above T 262144, the top of the pair-row single's measured range, for dw accuracy (above T 50176 the det pair's worst dw element exceeds the per-element budget at T 50688 on GB300, 1.032 of it); twosum: exact per-thread dw sums, so the dw error does not grow with T: worst dw element 0.0006 of the budget at the sampled T from 393216 to 1048576 on GB300 (3 x 120 inputs per T, kernel and torch rstd); the kblock single, 1.01 times its speed in the FI bench (GB300, 4 sessions), put its worst dw error at 1.36x the deterministic package's median FP64 error over 5 repeats at T 65536 on GB300 (concentrated inputs, one of three seeds): its 8-row FP32 blocks have the deterministic package's error, at block boundaries that change from run to run
    "cake_rmsnorm_train_8c8bebecc5b08454b6a0__262144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_8c8bebecc5b08454b6a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8c8bebecc5b08454b6a0",
        "role": "bwd_main",
        "select_rule": {
            "above": 262144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_107a H256 bf16 weight: (786432, inf] the twosum form with an L2 warm-up (172a47a1): exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above 36864 rows; it stays above 786432 rows, where the det pair exceeds the dw budget
    "cake_rmsnorm_train_172a47a15d4c1a76c061__786432_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "role": "bwd_main",
        "select_rule": {
            "above": 786432,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H256 f32 weight: (786432, inf] the twosum form with an L2 warm-up (7c2aa4b4): exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above 36864 rows; it stays above 786432 rows, where the det pair exceeds the dw budget
    "cake_rmsnorm_train_7c2aa4b429538da06a2a__786432_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 786432,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H4096 bf16 weight: (65536, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the single kernel it replaces exceeds the per-element budget at T 119040 (1.181) on R200 with 212 SMs, and so does the deterministic package's program for those token counts, at T 74610 (1.168, worst of 120 inputs of each of three families in both rstd forms); the window keeps its start at T 65536 until a follow-up has qualifying evidence for either program above it; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 65537 to 131072 (130 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.04x to 1.09x the single kernel's at T 65537, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_5e9d7baa36814bb7aaf6__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5e9d7baa36814bb7aaf6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5e9d7baa36814bb7aaf6",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_107a H4096 f32 weight: (24064, 65536] the deterministic package's compensated det pair of these rows (main kernel fa3d4fc5, the FP32-weight twin of 99a01f2a, and fixed-order dw reduce 10e24516 with that package's host plan), renamed into this package, for dw accuracy: the single kernel a1f537fc it replaces (FlashInfer e14ddfd9 runs the same one there) put its worst dw error at 1.340x, 1.244x and 1.547x the median FP64 error of the deterministic package at FlashInfer 2e989556 (its fa3d4fc5 pair) over 5 repeats at T 40960, 49152 and 65536 on R200 (3 seeds; concentrated, saturated_lo and concentrated inputs); the deterministic package at FlashInfer 2e989556 runs this pair at every T above 24064, the deterministic package now up to T 50176; its time and its worst dw error in this window are not measured yet
    "cake_rmsnorm_train_fa3d4fc5dbd76348b31a__24064_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fa3d4fc5dbd76348b31a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fa3d4fc5dbd76348b31a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24064,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H4096 f32 weight: (65536, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the single kernel it replaces exceeds the per-element budget at T 114422 (1.211) and the deterministic package's program for those token counts at T 88074 (1.050), worst of 120 inputs of each of three families in both rstd forms on R200 with 212 SMs; the window keeps its start at T 65536 until a follow-up has qualifying evidence for either program above it; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 65537 to 131072 (130 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.08x to 1.13x the single kernel's at T 65537, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_23cee954b648495ceef1__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_23cee954b648495ceef1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_23cee954b648495ceef1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_107a H6144 bf16 weight: (30720, 131072] the pair-row single kernel with its chunk table cut for 4 slots per SM (table_slots 4: more, shorter chunks than CTAs, the grid unchanged; twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on R200 with 212 SMs (CUDA graph replay, cold L2, 7 rounds) it took 110 to 434 us at T 30721 to 131072, 0.87x to 0.90x the time of the twosum single kernel 083b5f2e that FlashInfer main runs there (0.97x that of the deterministic package's program at T 32768 and 65536), with a dw error (relative L2 or worst element, the larger over six input families, 3 seeds x 5 repeats; dx equal to the deterministic package's) of at most 0.963x that of the deterministic package's program at FlashInfer 2e989556, equal to 083b5f2e's
    "cake_rmsnorm_train_3edefef1a5b4b9561a13__30720_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_3edefef1a5b4b9561a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3edefef1a5b4b9561a13",
        "role": "bwd_main",
        "select_rule": {
            "above": 30720,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_107a H6144 bf16 weight: (131072, inf] above the pair-row single's measured range (3edefef1 runs (30720, 131072]), the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the deterministic package's program for those token counts, faster here, exceeds the per-element budget at T 131072 (1.234) on R200 with 212 SMs, and its FP32-weight twin (the same dw computation) at T 98826 (1.098, worst of 120 inputs of each of three families in both rstd forms); the single kernel it replaces exceeds it at T 129472 (1.188, measured with FP32 weight; the BF16-weight program computes the same dw); its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 30721 to 131072 (65 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.78x to 0.81x the single kernel's at T 30721, 65536, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_107a H6144 f32 weight: (2304, 19456] the single kernel whose twosum form runs (28672, inf), for dw accuracy: the pair of (20, 28672] exceeded the FP32 eager reference's dw error at 18 sampled T from 4544 to 19264 on R200 with 212 SMs (the deterministic package runs the same pair there); its worst dw error stays at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges on R200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_e77387f14390ba486055__2304_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e77387f14390ba486055",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e77387f14390ba486055",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2304,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_107a H6144 f32 weight: (28672, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the single kernel it replaces exceeds the per-element budget at T 129472 (1.188) on R200 with 212 SMs, and so does the deterministic package's program for those token counts, faster here, at T 98826 (1.098, worst of 120 inputs of each of three families in both rstd forms); the window keeps its start at T 28672 until a follow-up has qualifying evidence for the deterministic package's program above it; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 28673 to 131072 (65 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.80x to 0.84x the single kernel's at T 28673, 65536, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_25f9b466fa418dd2afc4__28672_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_25f9b466fa418dd2afc4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_25f9b466fa418dd2afc4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 28672,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_103a H4096 f32 weight: (65536, 262144] above the det pairs of FlashInfer 2e989556 (faster up to T 65536 on GB300; not measured above); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.670 of the budget at 71 sampled T from 9217 to 131072 on GB300 (3 x 60 inputs per T, kernel and torch rstd), at most the FP32 eager reference's, 1.01 to 1.07 times the twosum single's speed in the FI bench (GB300, 4 sessions); up to T 262144, the largest sampled T at or below which its worst dw element stays within the per-element budget (0.933 of it); at T 393216 it is 1.038 of it (GB300, 3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_2911faf7a786144495a6__65536_262144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_2911faf7a786144495a6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2911faf7a786144495a6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_103a H4096 f32 weight: (32, 152] the deterministic package's column-task program of these token counts (128 column tasks), for speed: 0.84x to 0.88x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 33 to 152 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_11820fe48711775d5685__32_152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_11820fe48711775d5685",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_11820fe48711775d5685",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32,
            "at_most": 152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 f32 weight: (262144, inf] the twosum single above T 262144 for dw accuracy (the kblock single below exceeds the per-element budget at T 393216, 1.038 of it, on GB300); twosum: exact per-thread dw sums, so the dw error does not grow with T: worst dw element 0.0006 of the budget at the sampled T from 393216 to 1048576 on GB300 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_131a495c5897206b6874__262144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_131a495c5897206b6874",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_131a495c5897206b6874",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 262144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_103a H6144 bf16 weight: (24575, 65536] the pair-row single kernel with 96 helper CTAs and its chunk table cut for 6 slots per SM (table_slots 6: more, shorter chunks than CTAs, the grid unchanged) for speed: on GB300 (CUDA graph replay, cold L2, 7 rounds) it took 183.78 us and 351.71 us at T 32768 and 65536, 0.969x and 0.962x the time of the programs FlashInfer main runs there (the det pair 322574af, the kblock single 87e0b860), with a dw error (relative L2 or worst element, the larger over six input families) of at most 0.961x that of the deterministic package's program at FlashInfer 2e989556; from T 24576 to 32768, where the det pair 322574af ran before, its backward took 0.972x to 0.980x the det pair's time on GB300 (FI bench, L2 flushed before every call, two runs at T 24576, 28672 and 32768), with a dw error of at most 0.907x the median of the deterministic package's program at FlashInfer 2e989556 (six input families, three seeds, five repeats); below T 24576 the det pair stays: there a forward and backward call or a training step was slower with this kernel in both runs at some sampled T
    "cake_rmsnorm_train_f5806bd223b2ad27230a__24575_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 6,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_f5806bd223b2ad27230a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f5806bd223b2ad27230a",
        "role": "bwd_main",
        "select_rule": {
            "above": 24575,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_103a H6144 bf16 weight: (65536, 131072] the pair-row single kernel (twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on GB300 (CUDA graph replay, cold L2) it took 315 to 699 us at T 57345 to 131072, 0.97x to 0.99x the time of the programs FlashInfer e14ddfd9 runs at those T, with a dw error (relative L2 or worst element, the larger over six input families) of at most 0.972x that of the deterministic package's program at FlashInfer 2e989556
    "cake_rmsnorm_train_7236b7379272c82c7614__65536_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_7236b7379272c82c7614",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7236b7379272c82c7614",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 49152,
        },
    },
    # sm_103a H6144 bf16 weight: (131072, inf] the twosum single above T 131072: the kblock single's dw accuracy is qualified to T 131072 only (its worst dw element is 1.864 of the budget at T 1048576 on GB300)
    "cake_rmsnorm_train_582b2a20f9be648af8ec__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_582b2a20f9be648af8ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_582b2a20f9be648af8ec",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 f32 weight: (49152, 83392] the deterministic package's program as an ND pair for speed: backward 0.952-0.974 of the kblock single's time (R2-65 GB300 records); worst dw element 0.997 of the budget at the sampled T (GB300, at least 3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_aa46aa07d86900ad832f__49152_83392": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_aa46aa07d86900ad832f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_aa46aa07d86900ad832f",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 49152,
            "at_most": 83392,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__49152_83392": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 49152,
            "at_most": 83392,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H6144 f32 weight: (83392, 131072] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.028-1.044 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.699 of the budget at the sampled T up to 131072
    "cake_rmsnorm_train_935b8bea9166f29912ff__83392_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_935b8bea9166f29912ff",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_935b8bea9166f29912ff",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 83392,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 f32 weight: (131072, inf] the twosum single above T 131072: the kblock single's dw accuracy is qualified to T 131072 only (its worst dw element is 1.736 of the budget at T 1048576 on GB300)
    "cake_rmsnorm_train_4f9a76e68671f213ab7c__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_4f9a76e68671f213ab7c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4f9a76e68671f213ab7c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_107a H2048 bf16 weight: (50176, 50688] above T 50176 for dw accuracy (the det pair 845cdcd8's worst dw element reaches 0.900 of the per-element budget at T 51200 on R200 with 212 SMs; the deterministic package's compensated det pair 694decfb runs from T 50688 to 65536); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_1a3df5ac020055477581__50176_50688": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_1a3df5ac020055477581",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a3df5ac020055477581",
        "role": "bwd_main",
        "select_rule": {
            "above": 50176,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_107a H2048 bf16 weight: (50688, 65536] the deterministic package's det pair of these rows (its compensated program over 8-row blocks: main kernel 694decfb and fixed-order dw reduce c4d8be67 with that package's host plan, det record (50688, inf]), renamed into this package and cut to this window, for speed: on R200 with 212 SMs its backward took 0.831x to 0.854x the time of the twosum single 1a3df5ac at the sampled T from 50689 to 65536 (FI bench, L2 flushed before every call, two runs), its worst dw element stays within 0.58 of the per-element budget at T 50689, 57344 and 65536 (212 SMs, 45 inputs), and it is the program the deterministic package at FlashInfer 2e989556 runs there
    "cake_rmsnorm_train_694decfb79ba96468f62__50688_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_694decfb79ba96468f62",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_694decfb79ba96468f62",
        "role": "bwd_large",
        "select_rule": {
            "above": 50688,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c4d8be677fa70b841877__50688_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 50688,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H2048 bf16 weight: (65536, inf] above T 65536, the largest token count the dw gate against the deterministic package at FlashInfer 2e989556 samples on this route (the compensated det pair 694decfb runs below, from T 50688); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_1a3df5ac020055477581__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_1a3df5ac020055477581",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a3df5ac020055477581",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_107a H2048 f32 weight: (50176, inf] above T 50176 for dw accuracy (both weight dtypes switch at T 50176: the BF16-weight route's det pair, the same dw computation, reaches 0.900 of the per-element budget at T 51200 on R200 with 212 SMs; this route's det pair reaches 0.895 at T 57344 in the kernel-tests harness and 0.861 at T 61952 on R200 with 212 SMs); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_810989bdea464df479ce": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_810989bdea464df479ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_810989bdea464df479ce",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 50176,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_107a H2048 f32 weight: (1024, 16384] the single kernel of (50176, inf) for dw accuracy: the pair of the window exceeded the FP32 eager reference's dw error at T 1280 (torch rstd), 1728 and 1984 on R200 with 212 SMs (the deterministic package runs the same pair there); its worst dw error stays at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges on R200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_810989bdea464df479ce__1024_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_810989bdea464df479ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_810989bdea464df479ce",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_107a H1536 bf16 weight: (65536, inf] the twosum form of the single kernel (its knobs plus twosum and l2_ahead 2; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the deterministic package's program for those token counts, faster here, exceeds the per-element budget at T 91148 (1.026, worst of 120 inputs of each of three families in both rstd forms) and at T 124992 (1.098) on R200 with 212 SMs; the single kernel it replaces exceeds it at T 1048576 (2.259, worst of 120 inputs of each of three families in both rstd forms); the window keeps its start at T 65536 until a follow-up has qualifying evidence for the deterministic package's program above it; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 66624 to 131072 (54 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.79x to 0.80x the single kernel's at T 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_dca25ace64c075a81625__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_dca25ace64c075a81625",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dca25ace64c075a81625",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_107a H1536 f32 weight: (1024, 16384] the single kernel whose twosum form runs (65536, inf), for dw accuracy: the pair of the window exceeded the FP32 eager reference's dw error at T 1264 (torch rstd) on R200 with 212 SMs (the deterministic package runs the same pair there); its worst dw error stays at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges on R200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_7c0dd63f0b011049603e__1024_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7c0dd63f0b011049603e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c0dd63f0b011049603e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_107a H1536 f32 weight: (65536, inf] the twosum form of the single kernel (its knobs plus twosum and l2_ahead 2; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the deterministic package's program for those token counts, faster here, exceeds the per-element budget at T 106874 (1.018, worst of 120 inputs of each of three families in both rstd forms) and at T 131072 (1.119) on R200 with 212 SMs, and its BF16-weight twin (the same dw computation) at T 91148 (1.026); the single kernel it replaces exceeds it at T 118719 (1.049, worst of 120 inputs of each of three families in both rstd forms); the window keeps its start at T 65536 until a follow-up has qualifying evidence for the deterministic package's program above it; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 66624 to 131072 (54 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.79x to 0.80x the single kernel's at T 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_ccd94d87e9036a0b2faf__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_ccd94d87e9036a0b2faf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ccd94d87e9036a0b2faf",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_107a H512 f32 weight: (1024, 4096] for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 1088, 0.0896 against 0.0861 of the per-element budget, and at T 1664, 1920 and 2368 on R200 with 212 SMs; the det pair of (4096, 16384], run at these T, does too at T 1088 and 2048)
    "cake_rmsnorm_train_fd08c6f436eb6071fa22__1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
        },
    },
    # sm_100a H2048 bf16 weight: (58880, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in the shared row, evict_last loads, 3 stages): from T 58880 to 65536 for dw accuracy, it replaces the single kernel 0fa4f255, whose FP32 dw sums over windows of 16 steps put its worst dw error at 1.20x the median FP64 error of the det pair the deterministic package ran there at FlashInfer 2e989556 over 5 repeats at T 65536 on B200 (saturated_hi inputs); that det pair (0cb817d1 + d0960be9) reaches 0.883 of the per-element budget at T 59392 on B200, above this package's 0.85 det pair margin; above T 65536 it replaces the twosum single 83f905ff for speed up to T 131072; worst dw element 0.7531 of the budget at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T, kernel and torch rstd), not yet sampled from T 58881 to 65536; above T 131072, beyond the sampled token counts, the twosum single 83f905ff keeps the window; backward time 0.86 to 0.87 times the replaced single's above T 65536 (B200)
    "cake_rmsnorm_train_8b8cf4f5b82e77554908": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_8b8cf4f5b82e77554908",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8b8cf4f5b82e77554908",
        "role": "bwd_main",
        "select_rule": {
            "above": 58880,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 bf16 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) up to T 131072, the top of the qualified range (the single kernel it replaces stays within the per-element budget at every sampled T above T 65536, worst 0.887 at T 110016, on B200, worst of 60 inputs of each of three families in both rstd forms, not yet qualifying evidence); its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.10x to 1.13x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_83f905ffd00c0ef641e2__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_83f905ffd00c0ef641e2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_83f905ffd00c0ef641e2",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 f32 weight: (58880, 65536] the FP32-weight det pair the deterministic package runs here (main kernel 25068ede, the FP32-weight twin of 0cb817d1, and dw reduce d0960be9 with that package's host plan), renamed into this package, for dw accuracy at the single kernel's speed: the single kernel ba7dbf6a it replaces keeps within 1.1x the deterministic package's median dw error at T 65536 on B200 on the median rule only (off the gate's rows), and the det pair took 0.994x its backward time there
    "cake_rmsnorm_train_25068edea92269e1bdb1__58880_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_25068edea92269e1bdb1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_25068edea92269e1bdb1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 58880,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__58880_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 58880,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H2048 f32 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in the shared row, evict_last loads, 3 stages): it replaces the twosum single 9f775c72 for speed up to T 131072; worst dw element 0.7771 of the budget and at or below the FP32 eager reference's at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single 9f775c72 keeps the window; backward time 0.86 times the replaced single's (B200)
    "cake_rmsnorm_train_273d3f9b357669a821e2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_273d3f9b357669a821e2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_273d3f9b357669a821e2",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 f32 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) up to T 131072, the top of the qualified range (the single kernel it replaces stays within the per-element budget at every sampled T above T 65536, worst 0.942 at T 112240, on B200, worst of 60 inputs of each of three families in both rstd forms, not yet qualifying evidence); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.11x to 1.14x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_9f775c72c1335f4b08ac__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_9f775c72c1335f4b08ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9f775c72c1335f4b08ac",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    "cake_rmsnorm_train_7122dbcd697b3fbe1ee0__44032_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_7122dbcd697b3fbe1ee0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7122dbcd697b3fbe1ee0",
        "role": "bwd_main",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H4096 bf16 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; help finish, pairs in the shared row, evict_last loads, 3 stages): it replaces the twosum single c67fc8b9 for speed up to T 131072; worst dw element 0.8311 of the budget at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single c67fc8b9 keeps the window; backward time 0.96 to 0.97 times the replaced single's (B200)
    "cake_rmsnorm_train_fd7ae1d7547fca6156a0": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fd7ae1d7547fca6156a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd7ae1d7547fca6156a0",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 bf16 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) up to T 131072, the top of the qualified range (the worst dw element of the single kernel it replaces at the sampled T above T 65536 reaches 0.987 of the per-element budget, at T 120192, on B200, worst of 60 inputs of each of three families in both rstd forms, not yet qualifying evidence); its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.00x to 1.01x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_c67fc8b9d1b1edb8d7da__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_c67fc8b9d1b1edb8d7da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c67fc8b9d1b1edb8d7da",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 f32 weight: (44032, 65536] the FP32-weight det pair the deterministic package runs here (main kernel 5ef5f1f1, the FP32-weight twin of 7122dbcd, and dw reduce fcbfa14d with that package's host plan), renamed into this package, for dw accuracy: the single kernel f7573bcb, which FlashInfer e14ddfd9 runs here, put its worst dw element at 1.554x, 1.537x and 2.006x the deterministic package's (FlashInfer 2e989556) at T 45056, 49152 and 65536 on B200 (concentrated inputs; 5, 5 and 3 of 5 repeats above 1.1x), and the det pair took 0.990x, 0.997x and 0.978x its time there
    "cake_rmsnorm_train_5ef5f1f13ed8f2b994c4__44032_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5ef5f1f13ed8f2b994c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5ef5f1f13ed8f2b994c4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H4096 f32 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; help finish, pairs in the shared row, evict_last loads, 3 stages): it replaces the twosum single 9ebd0232 for speed up to T 131072; worst dw element 0.8588 of the budget and at or below the FP32 eager reference's at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single 9ebd0232 keeps the window; backward time 0.96 to 0.97 times the replaced single's (B200)
    "cake_rmsnorm_train_6e65495154249f51e74b": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_6e65495154249f51e74b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e65495154249f51e74b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 f32 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) up to T 131072, the top of the qualified range (the worst dw element of the single kernel it replaces at the sampled T above T 65536 reaches 0.907 of the per-element budget, at T 122112, on B200, worst of 60 inputs of each of three families in both rstd forms, not yet qualifying evidence); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 65537 to 131072 (82 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.00x to 1.01x the single kernel's at T 65537, 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_9ebd0232954f1b1842f7__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_9ebd0232954f1b1842f7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ebd0232954f1b1842f7",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_103a H2048 bf16 weight: (61952, 68160] the deterministic package's program as an ND pair for speed: backward 0.968-0.972 of the kblock single's time (R2-65 GB300 records); worst dw element 0.962 of the budget at the sampled T (GB300, at least 3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_018f0cd723106f709f11__61952_68160": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_018f0cd723106f709f11",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_018f0cd723106f709f11",
        "role": "bwd_main",
        "select_rule": {
            "above": 61952,
            "at_most": 68160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__61952_68160": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 61952,
            "at_most": 68160,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H2048 bf16 weight: (68160, 131072] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.082-1.084 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.782 of the budget at the sampled T up to 131072
    "cake_rmsnorm_train_2c32330b326a6db3f77b__68160_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_2c32330b326a6db3f77b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2c32330b326a6db3f77b",
        "role": "bwd_main",
        "select_rule": {
            "above": 68160,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H2048 bf16 weight: (131072, inf] the twosum single above T 131072: the kblock single's dw accuracy is qualified to T 131072 only (its worst dw element is 1.444 of the budget at T 1048576 on GB300)
    "cake_rmsnorm_train_8bd89a4c834f80db8a34__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_8bd89a4c834f80db8a34",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8bd89a4c834f80db8a34",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H2048 f32 weight: (61952, 68160] the deterministic package's program as an ND pair for speed: backward 0.969-0.974 of the kblock single's time (R2-65 GB300 records); worst dw element 0.962 of the budget at the sampled T (GB300, at least 3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_ca8ca4674a3dbd89ad16__61952_68160": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ca8ca4674a3dbd89ad16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ca8ca4674a3dbd89ad16",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 61952,
            "at_most": 68160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 f32 weight: (68160, 131072] kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs) for speed: backward 1.151-1.155 times as fast as the twosum single (R2-65 GB300 record); worst dw element 0.661 of the budget at the sampled T up to 131072
    "cake_rmsnorm_train_76e43a7d9c7ea1f59e0d__68160_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_76e43a7d9c7ea1f59e0d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_76e43a7d9c7ea1f59e0d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 68160,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H2048 f32 weight: (64, 171] the deterministic package's column-task program of these token counts (32 column tasks), for speed: 0.69x to 0.77x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 67 to 171 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_2e9171b7753650cda515__64_171": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_2e9171b7753650cda515",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2e9171b7753650cda515",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 f32 weight: (131072, inf] the twosum single above T 131072: the kblock single's dw accuracy is qualified to T 131072 only (its worst dw element is 1.383 of the budget at T 1048576 on GB300)
    "cake_rmsnorm_train_88c6c6b2af1922f4c1a9__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_88c6c6b2af1922f4c1a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_88c6c6b2af1922f4c1a9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 32768,
        },
    },
    "cake_rmsnorm_train_083ea823c691e0631ad3__97280_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "role": "bwd_main",
        "select_rule": {
            "above": 97280,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__97280_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 97280,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H1536 bf16 weight: (64, 384] the column-task form (16 columns per task): faster than the single kernel of (0, 14] and the det pair from 14 rows at every sampled T up to 384 on B200 in both sessions, level at 448 and slower from 512; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 14 on B200, 0.0048 against 0.0047 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_dc478ac1195882aedee6__64_384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_dc478ac1195882aedee6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dc478ac1195882aedee6",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H512 bf16 weight: (16, 227] the deterministic package's column-task program of this window (16 columns per task; twosum: exact per-thread dw sums): faster than the single kernel of (0, 26] and the pair of (26, 1024] it replaces at every sampled T from 1 to 227 in the balanced small-T sweep on B200 (lane RB, FI main) (their time 1.18x to 1.33x its); the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 26 on B200, 0.0062 against 0.0061 of the per-element budget)
    "cake_rmsnorm_train_5f3e3fccaea795d0a806__16_227": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_5f3e3fccaea795d0a806",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5f3e3fccaea795d0a806",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (64, 384] the column-task form (16 columns per task; twosum: exact per-thread dw sums): the single kernel of (0, 64] exceeded the FP32 eager reference's dw error at T 3 on B200 (torch rstd); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 1 to 1152 on B200, worst of 60 inputs of each of three families in both rstd forms, and it is faster than the programs it replaces up to T 384; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on B200, 0.0227 against 0.0203 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e__64_384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 f32 weight: (64, 1792] the column-task form (twosum: exact per-thread dw sums) for dw accuracy: the chunk-partials pairs of (22, 256] and (256, 6144] exceeded the FP32 eager reference's dw error at T 23 and 77 and at T 2752, 3840, 4224, 4352 and 5504 on B200 (the deterministic package runs the same pairs from T 122; up to T 121 it runs this kernel, with the same launch); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges (113 T) on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.34x to 1.27x the programs' it replaces at T 1, 4, 8, 13, 16, 22, 23, 77, 256, 1024, 1280, 1536 and 1792 in two sessions (B200); the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on B200, 0.0134 against 0.0130 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_68f6e584c687c30c0084__64_1792": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_68f6e584c687c30c0084",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_68f6e584c687c30c0084",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1792,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H4096 f32 weight: (16, 256] the det-new column-task program 440bd6f2 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.562x to 0.793x at the sampled T from 1 to 256, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 256] (120 inputs per T; 66 of them inside this window)
    "cake_rmsnorm_train_440bd6f2a114414ea9db__16_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_440bd6f2a114414ea9db",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_440bd6f2a114414ea9db",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 bf16 weight: (2, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.46x to 1.64x the det package's speed and 2.04x to 2.88x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 14 on B200, 0.0048 against 0.0047 of the per-element budget)
    "cake_rmsnorm_train_38e972a94141a1961c90__2_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_38e972a94141a1961c90",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e972a94141a1961c90",
        "role": "bwd_main",
        "select_rule": {
            "above": 2,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (2, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.40x to 1.68x the det package's speed and 2.49x to 3.06x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on B200, 0.0227 against 0.0203 of the per-element budget)
    "cake_rmsnorm_train_d5a621af7a24e403b5f3__2_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_d5a621af7a24e403b5f3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d5a621af7a24e403b5f3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 f32 weight: (4, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.16x to 1.41x the det package's speed and 1.71x to 3.79x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on B200, 0.0134 against 0.0130 of the per-element budget)
    "cake_rmsnorm_train_244b7d841d62d9721c99__4_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_244b7d841d62d9721c99",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_244b7d841d62d9721c99",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (5, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.31x to 1.59x the det package's speed and 1.44x to 3.77x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 14 on GB300, 0.0048 against 0.0047 of the per-element budget)
    "cake_rmsnorm_train_38e972a94141a1961c90__0_64_103": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_38e972a94141a1961c90",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e972a94141a1961c90",
        "role": "bwd_main",
        "select_rule": {
            "above": 5,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 bf16 weight: (16, 63] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.17x to 1.62x the det package's speed and 1.24x to 4.56x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 12 on GB300, 0.0047 against 0.0045 of the per-element budget)
    "cake_rmsnorm_train_ef052c0918725ffa68d4__16_63": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ef052c0918725ffa68d4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ef052c0918725ffa68d4",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 63,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 bf16 weight: (63, 171] the deterministic package's column-task program of these token counts (32 column tasks), for speed: 0.69x to 0.86x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 64 to 171 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_2b4c0f5b0e9c312fa43e__63_171": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_2b4c0f5b0e9c312fa43e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2b4c0f5b0e9c312fa43e",
        "role": "bwd_main",
        "select_rule": {
            "above": 63,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H256 bf16 weight: (64, 1080] rows the deterministic package's det pair of these rows (main kernel 083ea823 and fixed-order dw reduce b4b63d5b with that package's host plan), renamed into this package, for dw accuracy: the single kernel it replaces rounds every row's dw product to FP32, which put its worst dw error at up to 3.46x the det pair's median FP64 error over 5 repeats at T 1 of the [T, 4, 256] and [T, 24, 256] views on GB300 (FP32 weight; the BF16-weight single computes the same dw); the det pair also took 0.84x to 0.88x the single's backward time there (GB300, CUDA-graph span, 4 sessions)
    "cake_rmsnorm_train_083ea823c691e0631ad3__64_1080": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H5120 bf16 weight: (4, 16] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.03x to 1.44x the det package's speed and 1.02x to 1.44x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on GB300, 0.0134 against 0.0119 of the per-element budget)
    "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__4_16": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (6, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.33x to 1.57x the det package's speed and 2.73x to 3.69x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 320 on GB300, 0.0596 against 0.0583 of the per-element budget, and at T 64, 0.0227 against 0.0203)
    "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64_103": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_d5a621af7a24e403b5f3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d5a621af7a24e403b5f3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 6,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (64, 214] the deterministic package's column-task program of these token counts (32 column tasks), for speed: 0.38x to 0.42x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 65 to 214 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_5c5be9a6aeac20fd2776__64_214": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_5c5be9a6aeac20fd2776",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5c5be9a6aeac20fd2776",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 f32 weight: (16, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.23x to 1.57x the det package's speed and 1.21x to 4.32x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 26 on GB300, 0.0152 against 0.0134 of the per-element budget, and at T 42 in the kernel-tests harness, 0.0205 against 0.0189)
    "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64_103": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_382e45788ace3ff40ba1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_382e45788ace3ff40ba1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H256 f32 weight: (64, 1080] rows the deterministic package's det pair of these rows (main kernel c4c89f0d and fixed-order dw reduce b4b63d5b with that package's host plan), renamed into this package, for dw accuracy: the single kernel it replaces rounds every row's dw product to FP32, which put its worst dw error at up to 3.46x the det pair's median FP64 error over 5 repeats at T 1 of the [T, 4, 256] and [T, 24, 256] views on GB300 (FP32 weight); the det pair also took 0.84x to 0.88x the single's backward time there (GB300, CUDA-graph span, 4 sessions)
    "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__64_1080": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 f32 weight: (16, 32] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.28x to 1.44x the det package's speed and 1.40x to 4.45x the programs it replaces at every sampled T from 1 to 32 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 15 on GB300, 0.0105 against 0.0092 of the per-element budget)
    "cake_rmsnorm_train_c1b9ee6b9f10766cd537__16_32": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_c1b9ee6b9f10766cd537",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c1b9ee6b9f10766cd537",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H5120 f32 weight: (5, 16] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.03x to 1.47x the det package's speed and 1.03x to 5.03x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on GB300, 0.0134 against 0.0130 of the per-element budget)
    "cake_rmsnorm_train_244b7d841d62d9721c99__5_16": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_244b7d841d62d9721c99",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_244b7d841d62d9721c99",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 5,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 f32 weight: (12, 16] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.03x to 1.53x the det package's speed and 1.04x to 1.53x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 20 on GB300, 0.0147 against 0.0138 of the per-element budget)
    "cake_rmsnorm_train_9b37923697ee8ca52a9e__12_16": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_9b37923697ee8ca52a9e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9b37923697ee8ca52a9e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (10, 31] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.08x to 1.55x the det package's speed and 1.07x to 1.92x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 6 on GB300, 0.0067 against 0.0063 of the per-element budget)
    "cake_rmsnorm_train_ac9d69f94aabfc68aa40__10_31": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 10,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (31, 32] the deterministic package's column-task program of these token counts (96 column tasks), for speed: 0.81x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 17 to 32 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_570eec506ed94f36f062__31_32": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_570eec506ed94f36f062",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_570eec506ed94f36f062",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 31,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (32, 48] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.08x to 1.55x the det package's speed and 1.07x to 1.92x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 6 on GB300, 0.0067 against 0.0063 of the per-element budget)
    "cake_rmsnorm_train_ac9d69f94aabfc68aa40__32_48": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 32,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 bf16 weight: (16, 320] the det-new column-task program 9e71ed0d of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.660x to 0.941x at the sampled T from 16 to 320, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 576] (120 inputs per T; 35 of them inside this window)
    "cake_rmsnorm_train_9e71ed0d9ad7a7739002__16_320": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9e71ed0d9ad7a7739002",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e71ed0d9ad7a7739002",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 bf16 weight: (16, 31] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.14x to 1.52x the det package's speed and 1.14x to 3.00x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 6 on R200 with 212 SMs, 0.0028 against 0.0027 of the per-element budget)
    "cake_rmsnorm_train_ef052c0918725ffa68d4__16_31": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ef052c0918725ffa68d4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ef052c0918725ffa68d4",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 bf16 weight: (31, 256] the det-new column-task program 32eb12e2 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.648x to 0.928x at the sampled T from 32 to 256, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 256] (120 inputs per T; 58 of them inside this window)
    "cake_rmsnorm_train_32eb12e27be1360b890d__31_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_32eb12e27be1360b890d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32eb12e27be1360b890d",
        "role": "bwd_main",
        "select_rule": {
            "above": 31,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H256 bf16 weight: (64, 786432] rows the det-new pair (#8813) of the deterministic package, main kernel 083ea823 and fixed-order dw reduce b4b63d5b, renamed into this package (the copies sm_100a and sm_103a already run); faster than the ND program at every sampled row count up to 786432 in two R200 records with 212 SMs (30 row counts of the flat, per-head [T, 24, 256] and [T, 4, 256] views, det/ND 0.806x to 0.952x over both records) and in a VR200 screen; worst dw element 0.713 of the per-element budget up to 786432 rows (120 inputs x 3 families per sampled row count), 0.723 in an earlier flat and [T, 24, 256] run above 36864 rows; above 786432 rows its FP32-weight twin exceeds the dw budget (1.057 at 1271400 rows) and this pair computes the same dw, so the ND program stays there
    "cake_rmsnorm_train_083ea823c691e0631ad3__64_786432": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_083ea823c691e0631ad3",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 786432,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b4b63d5b5028c7a60333__64_786432": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 786432,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H256 f32 weight: (64, 786432] rows the det-new pair (#8813) of the deterministic package, main kernel c4c89f0d and fixed-order dw reduce b4b63d5b, renamed into this package (the copies sm_100a and sm_103a already run); faster than the ND program at every sampled row count up to 786432 in two R200 records with 212 SMs (30 row counts of the flat, per-head [T, 24, 256] and [T, 4, 256] views, det/ND 0.761x to 0.946x over both records) and in a VR200 screen; worst dw element 0.713 of the per-element budget up to 786432 rows (120 inputs x 3 families per sampled row count), 0.723 in an earlier flat and [T, 24, 256] run above 36864 rows; above 786432 rows this pair exceeds the dw budget (1.057 at 1271400 rows), so the ND program stays there
    "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__64_786432": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 786432,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H512 f32 weight: (16, 384] the det-new column-task program 685b53bf of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.547x to 0.889x at the sampled T from 1 to 384, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 768] (120 inputs per T; 33 of them inside this window)
    "cake_rmsnorm_train_685b53bf6ce1b6ff13f8__16_384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_685b53bf6ce1b6ff13f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_685b53bf6ce1b6ff13f8",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (1664, 2304] the column-task form (twosum: exact per-thread dw sums); from T 20 for dw accuracy: the pair of (20, 28672] exceeded the FP32 eager reference's dw error at 18 sampled T from 4544 to 19264 on R200 with 212 SMs (the deterministic package runs the same pair there); its worst dw error stays at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges on R200, worst of 60 inputs of each of three families in both rstd forms; faster than the single kernel of (2304, 19456] up to T 2304; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 20 on R200 with 212 SMs, 0.0147 against 0.0138 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_44c5b870a7dcbb8de924__1664_2304": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_44c5b870a7dcbb8de924",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44c5b870a7dcbb8de924",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1664,
            "at_most": 2304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H4096 bf16 weight: (16, 256] the det-new column-task program 44511fa0 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.619x to 0.868x at the sampled T from 2 to 256, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 256] (120 inputs per T; 65 of them inside this window)
    "cake_rmsnorm_train_44511fa02465a6d7283d__16_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_44511fa02465a6d7283d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44511fa02465a6d7283d",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 f32 weight: (16, 320] the det-new column-task program db8d986d of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.607x to 0.919x at the sampled T from 2 to 320, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 576] (120 inputs per T; 36 of them inside this window)
    "cake_rmsnorm_train_db8d986d719b909621d0__16_320": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db8d986d719b909621d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_db8d986d719b909621d0",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 f32 weight: (16, 256] the det-new column-task program 22944704 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.574x to 0.915x at the sampled T from 1 to 256, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 256] (120 inputs per T; 66 of them inside this window)
    "cake_rmsnorm_train_22944704b77a80a04834__16_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_22944704b77a80a04834",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_22944704b77a80a04834",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (16, 1664] the det-new column-task program a501b31f of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.650x to 0.945x at the sampled T from 2 to 1664, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1664] (120 inputs per T; 65 of them inside this window)
    "cake_rmsnorm_train_a501b31feb94fa326722__16_1664": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a501b31feb94fa326722",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a501b31feb94fa326722",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 1664,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (16, 768] the det-new column-task program a1c12b0e of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.630x to 0.973x at the sampled T from 1 to 768, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 960] (120 inputs per T; 53 of them inside this window)
    "cake_rmsnorm_train_a1c12b0ee6454391a37f__16_768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 48,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a1c12b0ee6454391a37f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a1c12b0ee6454391a37f",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 16,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 bf16 weight: (16, 32] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.27x to 1.47x the det package's speed and 1.42x to 4.61x the programs it replaces at every sampled T from 1 to 32 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 3 on GB300, by less than 0.0001 of the per-element budget)
    "cake_rmsnorm_train_3dbab2323fa42435a72d__16_32": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_3dbab2323fa42435a72d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3dbab2323fa42435a72d",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (6, 16] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.20x to 1.51x the det package's speed and 1.19x to 1.95x the programs it replaces at every sampled T from 1 to 32 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 3 on GB300, by less than 0.0001 of the per-element budget)
    "cake_rmsnorm_train_d70f2562a592151ec4f9__6_16": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_d70f2562a592151ec4f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d70f2562a592151ec4f9",
        "role": "bwd_main",
        "select_rule": {
            "above": 6,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (16, 960] the det-new column-task program 57d45549 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.639x to 0.967x at the sampled T from 16 to 960, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1664] (120 inputs per T; 37 of them inside this window)
    "cake_rmsnorm_train_57d45549f8ed34e34e65__16_960": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_57d45549f8ed34e34e65",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_57d45549f8ed34e34e65",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (1151, 1152] the det-new column-task program 57d45549 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.916x at T 1152, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1664] (120 inputs per T; 1 of them inside this window)
    "cake_rmsnorm_train_57d45549f8ed34e34e65__1151_1152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_57d45549f8ed34e34e65",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_57d45549f8ed34e34e65",
        "role": "bwd_main",
        "select_rule": {
            "above": 1151,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (15, 320] the det-new column-task program c15868b8 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.600x to 0.938x at the sampled T from 16 to 320, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 960] (120 inputs per T; 20 of them inside this window)
    "cake_rmsnorm_train_c15868b87bfb2c21645e__15_320": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 48,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c15868b87bfb2c21645e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c15868b87bfb2c21645e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 15,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (447, 576] the det-new column-task program c15868b8 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.867x to 0.897x at the sampled T from 448 to 576, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 960] (120 inputs per T; 9 of them inside this window)
    "cake_rmsnorm_train_c15868b87bfb2c21645e__447_576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 48,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c15868b87bfb2c21645e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c15868b87bfb2c21645e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 447,
            "at_most": 576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H512 bf16 weight: (16, 64] the deterministic package's column-task program of these token counts (32 column tasks), for speed: 0.50x to 0.80x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 5 to 62 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_877751ba915bbc261064__16_64": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_877751ba915bbc261064",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_877751ba915bbc261064",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H512 bf16 weight: (16, 576] the det-new column-task program bcd8d6a9 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.528x to 0.934x at the sampled T from 1 to 576, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 768] (120 inputs per T; 49 of them inside this window)
    "cake_rmsnorm_train_bcd8d6a9c739c7a11c36__16_576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_bcd8d6a9c739c7a11c36",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bcd8d6a9c739c7a11c36",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (49152, inf] above T 49152, where the det pair stops; kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs in registers): worst dw element 0.807 of the budget at 265 sampled T from 49153 to 131072 on R200 (212 SMs, 3 x 60 inputs per T, kernel and torch rstd), 1.74 to 1.87 times the FP64-row single's speed in the FI bench (R200, 4 sessions)
    "cake_rmsnorm_train_56634f53e6456b34f76f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_56634f53e6456b34f76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56634f53e6456b34f76f",
        "role": "bwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_107a H5120 f32 weight: (32767, 131072] the pair-row single kernel in its kblock form (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs in registers, pipe) with its chunk table cut for 4 slots per SM (table_slots 4: more, shorter chunks than CTAs, the grid unchanged; each CTA writes its pairs as one row of a scratch area, and 128 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on R200 with 212 SMs (CUDA graph replay, cold L2, 7 rounds) it took 99.6 to 363 us at T 32768 to 131072, 0.93x to 0.95x the time of the det pair de0cf457 + cdcd68d9 (up to T 49152) and of the kblock single kernel 6cedccdf (above) it replaces, with a dw error (relative L2 or worst element, the larger over six input families, 3 seeds x 5 repeats; dx equal to the deterministic package's) of at most 1.08x that of the deterministic package's program at FlashInfer 2e989556; at T 20480 it reaches 1.13x, so the det pair keeps the window below; only the FP32-weight twin ships
    "cake_rmsnorm_train_9cdd6eef6888b2a1dd6a__32767_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9cdd6eef6888b2a1dd6a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9cdd6eef6888b2a1dd6a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32767,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 40960,
        },
    },
    # sm_107a H5120 f32 weight: (131072, inf] above the pair-row single's measured range (9cdd6eef runs (32767, 131072]); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs in registers): worst dw element 0.719 of the budget at 265 sampled T from 49153 to 131072 on R200 (212 SMs, 3 x 60 inputs per T, kernel and torch rstd), at most the FP32 eager reference's, 1.66 to 1.78 times the FP64-row single's speed in the FI bench (R200, 4 sessions)
    "cake_rmsnorm_train_6cedccdf3a58f23f260f__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_6cedccdf3a58f23f260f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6cedccdf3a58f23f260f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_100a H512 f32 weight: (16, 227] the deterministic package's column-task program of this window (16 columns per task; twosum: exact per-thread dw sums): faster than the single kernel of (0, 292] it replaces at every sampled T from 1 to 227 in the balanced small-T sweep on B200 (lane RB, FI main) (its time 1.30x to 1.41x); the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 288 on B200, 0.0521 against 0.0409 of the per-element budget, and the FP64-accumulating single kernel this route runs above T 960 exceeds it at T 4, 0.0033 against 0.0032; with the kernel-tests harness inputs included the det pair's worst dw error exceeds it at T 292, 0.0558 against 0.0554)
    "cake_rmsnorm_train_942503b80a21f22cd392__16_227": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_942503b80a21f22cd392",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_942503b80a21f22cd392",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H512 f32 weight: (227, 292] for dw accuracy at small T (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 288 on B200, 0.0521 against 0.0409 of the per-element budget, and the FP64-accumulating single kernel this route runs above T 960 exceeds it at T 4, 0.0033 against 0.0032; with the kernel-tests harness inputs included the det pair's worst dw error exceeds it at T 292, 0.0558 against 0.0554); twosum: exact per-thread dw sums; the deterministic package's column-task program runs up to T 227
    "cake_rmsnorm_train_aa7c5a5407cba063b9d5__227_292": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_aa7c5a5407cba063b9d5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_aa7c5a5407cba063b9d5",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 227,
            "at_most": 292,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
        },
    },
    # sm_107a H5120 bf16 weight: (12, 31] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.29x to 1.71x the det package's speed and 1.29x to 3.56x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 15 on R200 with 212 SMs, 0.0120 against 0.0114 of the per-element budget, and at T 18 in the kernel-tests harness, 0.0136 against 0.0134)
    "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__12_31": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c",
        "role": "bwd_main",
        "select_rule": {
            "above": 12,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (31, 256] the det-new column-task program 505283a5 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.596x to 0.847x at the sampled T from 32 to 256, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1280] (120 inputs per T; 12 of them inside this window)
    "cake_rmsnorm_train_505283a5225ac45bc095__31_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_505283a5225ac45bc095",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_505283a5225ac45bc095",
        "role": "bwd_main",
        "select_rule": {
            "above": 31,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (511, 640] the det-new column-task program 505283a5 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.911x to 0.928x at the sampled T from 512 to 640, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1280] (120 inputs per T; 7 of them inside this window)
    "cake_rmsnorm_train_505283a5225ac45bc095__511_640": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_505283a5225ac45bc095",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_505283a5225ac45bc095",
        "role": "bwd_main",
        "select_rule": {
            "above": 511,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (895, 896] the det-new column-task program 505283a5 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.953x at T 896, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1280] (120 inputs per T; none of them inside this window)
    "cake_rmsnorm_train_505283a5225ac45bc095__895_896": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_505283a5225ac45bc095",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_505283a5225ac45bc095",
        "role": "bwd_main",
        "select_rule": {
            "above": 895,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (15, 256] the det-new column-task program 3a7bd162 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.605x to 0.930x at the sampled T from 16 to 256, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1280] (120 inputs per T; 13 of them inside this window)
    "cake_rmsnorm_train_3a7bd162c370762719b3__15_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 15,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (511, 640] the det-new column-task program 3a7bd162 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.900x to 0.915x at the sampled T from 512 to 640, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1280] (120 inputs per T; 7 of them inside this window)
    "cake_rmsnorm_train_3a7bd162c370762719b3__511_640": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 511,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (895, 896] the det-new column-task program 3a7bd162 of the deterministic package (#8813), renamed into this package; faster than the ND program at every sampled T in two R200 records with 212 SMs (det/ND 0.935x at T 896, the slower record per T) and in a VR200 screen; worst dw element 0.0006 of the per-element budget over 66 sampled T of the program's window (0, 1280] (120 inputs per T; none of them inside this window)
    "cake_rmsnorm_train_3a7bd162c370762719b3__895_896": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 895,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H512 f32 weight: (16, 227] the deterministic package's column-task program of these token counts (32 column tasks), for speed: 0.39x to 0.42x the backward time of the programs it replaces at the sampled T in two GB300 records; worst dw element 0.0006 of the budget at the sampled T from 5 to 227 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_81d9adbf31a26a4ebc02__16_227": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_81d9adbf31a26a4ebc02",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_81d9adbf31a26a4ebc02",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_7dc29629ff21625607f8__384_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The deterministic package's backward pairs (main kernel, then its dw reduce), renamed
    # into this package, for the windows the routes run them in.
    "cake_rmsnorm_train_7dc29629ff21625607f8__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c1a9efdf06c3087b4efe": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_21c8bb55c8d672b63952": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_21c8bb55c8d672b63952",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_21c8bb55c8d672b63952",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b457c4d134e44476dc6f__32768_36352": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_b457c4d134e44476dc6f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b457c4d134e44476dc6f",
        "role": "bwd_large",
        "select_rule": {
            "above": 32768,
            "at_most": 36352,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b5bdabba0b80fb8afcc5__36352_78784": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_b5bdabba0b80fb8afcc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5bdabba0b80fb8afcc5",
        "role": "bwd_large",
        "select_rule": {
            "above": 36352,
            "at_most": 78784,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_36352": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 36352,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__36352_78784": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 36352,
            "at_most": 78784,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__384_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_17d9cdfd157ba4b86c2b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_4bf450b6af0314ed03be": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4bf450b6af0314ed03be",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4bf450b6af0314ed03be",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__336_3072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "role": "bwd_small",
        "select_rule": {
            "above": 336,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9ac8d21e46a658e7913f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_9ac8d21e46a658e7913f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ac8d21e46a658e7913f",
        "role": "bwd_mid",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c73074cf44f07fcf3bb6__4096_7168": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "role": "bwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H2048 bf16 weight: (11264, 16384] the deterministic package's det pair of these rows (main kernel c73074cf and fixed-order dw reduce d0960be9 with that package's host plan), renamed into this package; from T 13608 for dw accuracy: the single kernel 0fa4f255 it replaces there sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.26x to 1.62x the deterministic package's median FP64 error over 5 repeats at T 14928 to 16384 on B200 (concentrated inputs); the det pair took 38.02, 40.38 and 40.32 us against the single's 36.48, 38.77 and 38.88 us at T 14928, 16172 and 16384 (B200, CUDA-graph span, 2 sessions)
    "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "role": "bwd_large",
        "select_rule": {
            "above": 11264,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 336,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__4096_7168": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__11264_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 11264,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__16384_58880": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d96a14fe8882bbd9ec2e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d96a14fe8882bbd9ec2e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d96a14fe8882bbd9ec2e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b187e5989302def22b2d__4096_15403": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b187e5989302def22b2d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b187e5989302def22b2d",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 15403,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__4096_15403": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 15403,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9a830956447d4e304914__1152_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_9a830956447d4e304914",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a830956447d4e304914",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_4608": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_44032": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 36864,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__1152_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_4608": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 36864,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__44032_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_626aed743358c27513f1__36864_44032": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 36864,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_09601d42628de24f6f87__64_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_09601d42628de24f6f87",
        "role": "bwd_small",
        "select_rule": {
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_09601d42628de24f6f87__227_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_09601d42628de24f6f87",
        "role": "bwd_small",
        "select_rule": {
            "above": 227,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_123db0b2f7fb38cfab03__1024_1280": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_123db0b2f7fb38cfab03",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_123db0b2f7fb38cfab03",
        "role": "bwd_low",
        "select_rule": {
            "above": 1024,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__64_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 227,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__4096_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_122721bd6b4650866553__32768_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fef03ce50a792c8704bd__1024_1280": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_fef03ce50a792c8704bd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fef03ce50a792c8704bd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9f07cea1d13b923ed91e__292_960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 292,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H512 f32 weight: (227, 960] the det pair; from T 227 to 292, where it replaces a single kernel, for speed: 0.50x that kernel's backward time at the sampled T in two GB300 records; worst dw element 0.0575 of the budget at the sampled T from 228 to 292 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_9f07cea1d13b923ed91e__227_960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 227,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__292_960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 292,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 227,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b482c84dbffcd75a4a4c__896_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b482c84dbffcd75a4a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b482c84dbffcd75a4a4c",
        "role": "bwd_main",
        "select_rule": {
            "above": 896,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c145ef195d7aeb04f242__6144_56256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c145ef195d7aeb04f242",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c145ef195d7aeb04f242",
        "role": "bwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": 56256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H5120 bf16 weight: (56256, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in registers, evict_last loads, 4 stages): it replaces the twosum single ade9f6a4 for speed up to T 131072; worst dw element 0.8835 of the budget at the sampled T from 56257 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single ade9f6a4 keeps the window; backward time 0.80 to 0.82 times the replaced single's (B200)
    "cake_rmsnorm_train_56634f53e6456b34f76f__56256_131072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_56634f53e6456b34f76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56634f53e6456b34f76f",
        "role": "bwd_main",
        "select_rule": {
            "above": 56256,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_100a H5120 bf16 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the det pair it replaces exceeds the per-element budget at T 56320 (1.024; worst of 240 inputs of each of three families in both rstd forms, every 64th T from 45056 to 57344; at most 0.842 up to T 56256) on B200; the single kernel it replaces above T 65536 exceeds the budget at T 125952 (1.040, measured with FP32 weight; the BF16-weight program computes the same dw) on B200; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 57345 to 131072 (82 T) and within 0.0006 of it at T 56257, 56320, 56832 and 57344 on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.24x to 1.24x the single kernel's at T 57345 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_ade9f6a4f10b069ba8a3__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_ade9f6a4f10b069ba8a3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ade9f6a4f10b069ba8a3",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__896_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 896,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__6144_56256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 6144,
            "at_most": 56256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9479b1298a1b19d4d996__6144_56256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_9479b1298a1b19d4d996",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9479b1298a1b19d4d996",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 56256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H5120 f32 weight: (56256, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in registers, evict_last loads, 4 stages): it replaces the twosum single fc2a52a2 for speed up to T 131072; worst dw element 0.9988 of the budget and at or below the FP32 eager reference's at the sampled T from 56257 to 131072 on B200 (3 x 120 inputs per T and 3 x 240 at 40 T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single fc2a52a2 keeps the window; backward time 0.79 to 0.81 times the replaced single's (B200)
    "cake_rmsnorm_train_6cedccdf3a58f23f260f__56256_131072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_6cedccdf3a58f23f260f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6cedccdf3a58f23f260f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 56256,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_100a H5120 f32 weight: (131072, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the det pair it replaces is the FP32-weight twin of the BF16-weight route's det pair (the same dw for the same inputs), which exceeds the per-element budget at T 56320 (1.024) on B200; the single kernel it replaces above T 65536 exceeds the budget at T 125952 (1.040) on B200; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 57345 to 131072 (82 T) and within 0.0006 of it, at or below the FP32 eager reference's, at T 56257, 56320, 56832 and 57344 on B200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.13x to 1.13x the programs it replaces at T 98304 and 131072 in two sessions (B200); above T 131072 it also stands in for the kblock single of the window below, whose dw accuracy is sampled only up to T 131072 (B200)
    "cake_rmsnorm_train_fc2a52a2c9bf8c5be9af__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_fc2a52a2c9bf8c5be9af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fc2a52a2c9bf8c5be9af",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "role": "bwd_low",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H6144 bf16 weight: (4096, 42496] the deterministic package's det pair of these rows (main kernel b93bf060 and fixed-order dw reduce 5f5cdc51 with that package's host plan), renamed into this package; from T 5632 for dw accuracy: the single kernel 9433061d it replaces there sums each thread's dw in FP32 over windows of 16 steps, which put its worst dw error at 1.10x to 1.62x the deterministic package's median FP64 error over 5 repeats at T 14928 to 16231 on B200 (concentrated and saturated_lo inputs); the det pair took 95.26, 102.08 and 102.69 us against the single's 93.22, 99.90 and 100.74 us at T 14928, 16172 and 16384 (B200, CUDA-graph span, 2 sessions)
    "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_b93bf060d4d3defc8150",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b93bf060d4d3defc8150",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_41b80b6af97974feb181__1168_3584": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41b80b6af97974feb181",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 1168,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1168_3584": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1168,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1df7db17277d363b5168": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1df7db17277d363b5168",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1df7db17277d363b5168",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 14928,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_4df08ce51fb01ba1193f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4df08ce51fb01ba1193f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4df08ce51fb01ba1193f",
        "role": "bwd_large",
        "select_rule": {
            "above": 14928,
            "at_most": 15566,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_f522f5b8d4343d96ca04__15566_46592": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_f522f5b8d4343d96ca04",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f522f5b8d4343d96ca04",
        "role": "bwd_large",
        "select_rule": {
            "above": 15566,
            "at_most": 46592,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 14928,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 14928,
            "at_most": 15566,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__15566_46592": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 15566,
            "at_most": 46592,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 f32 weight: (214, 1024] the det pair; from T 214 to 320, where it replaces a single kernel, for speed: 0.40x to 0.53x that kernel's backward time at the sampled T in two GB300 records; worst dw element 0.0596 of the budget at the sampled T from 215 to 320 (3 x 120 inputs per T, kernel and torch rstd)
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_a99e2b380514c4d86d60": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_a99e2b380514c4d86d60",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a99e2b380514c4d86d60",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 14928,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 14928,
            "at_most": 15566,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_109f19bb177c8f20cadc__15566_37888": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_109f19bb177c8f20cadc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_109f19bb177c8f20cadc",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 15566,
            "at_most": 37888,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__15566_37888": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 15566,
            "at_most": 37888,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__171_3072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "role": "bwd_small",
        "select_rule": {
            "above": 171,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__large": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "role": "bwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 bf16 weight: (32768, 58496] the det pair up to T 58496: above it the pair breaches the per-element dw budget (worst dw element 1.0622 of the budget at T 58560 on GB300, FP32-weight route, torch rstd, 3 x 120 and 3 x 240 inputs; the BF16-weight route runs the same dw computation), so the window ends at the last every-64 T below that point
    "cake_rmsnorm_train_302f64db6abe79e5259e__32768_58496": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": 58496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 58496,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H2048 bf16 weight: (58496, 61952] the deterministic package's program for these token counts for dw accuracy up to T 61952: worst dw element 0.9454 of the budget at the sampled T from 58497 to 61952 on GB300 (3 x 120 inputs per T, kernel and torch rstd); backward time 1.01 of the det pair's (GB300 record)
    "cake_rmsnorm_train_018f0cd723106f709f11__58496_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_018f0cd723106f709f11",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_018f0cd723106f709f11",
        "role": "bwd_main",
        "select_rule": {
            "above": 58496,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 58496,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_880768a0a264e04dbfb5__171_3072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 171,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_596b457aaba073f8d69f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_596b457aaba073f8d69f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_596b457aaba073f8d69f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_07f9f2fbea7163639c27__171_3072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 171,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_07f9f2fbea7163639c27__large": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_07f9f2fbea7163639c27__16384_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 f32 weight: (32768, 58496] the det pair up to T 58496: above it the pair breaches the per-element dw budget (worst dw element 1.0622 of the budget at T 58560 on GB300, FP32-weight route, torch rstd, 3 x 120 and 3 x 240 inputs; the BF16-weight route runs the same dw computation), so the window ends at the last every-64 T below that point
    "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_58496": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 58496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 f32 weight: (58496, 61952] the deterministic package's program for these token counts for dw accuracy up to T 61952: worst dw element 0.9454 of the budget at the sampled T from 58497 to 61952 on GB300 (3 x 120 inputs per T, kernel and torch rstd); backward time 1.01 of the det pair's (GB300 record)
    "cake_rmsnorm_train_ca8ca4674a3dbd89ad16__58496_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_ca8ca4674a3dbd89ad16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ca8ca4674a3dbd89ad16",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 58496,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_af005462cd6a301f31c4__1080_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_af005462cd6a301f31c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_af005462cd6a301f31c4",
        "role": "bwd_main",
        "select_rule": {
            "above": 1080,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1080,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_a385ae1681a2c078344e__5120_12288": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_af005462cd6a301f31c4__12288_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_af005462cd6a301f31c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_af005462cd6a301f31c4",
        "role": "bwd_main",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_a385ae1681a2c078344e__40960_97280": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 40960,
            "at_most": 97280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0291949ba535c646c3a8__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_0291949ba535c646c3a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0291949ba535c646c3a8",
        "role": "bwd_main",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c18297d75c0d424978fd": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_c18297d75c0d424978fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c18297d75c0d424978fd",
        "role": "bwd_main",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
            "partial_rows": 2,
        },
    },
    "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__12288_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__40960_97280": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 40960,
            "at_most": 97280,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b4b63d5b5028c7a60333__64_1080": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b4b63d5b5028c7a60333__97280_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_b4b63d5b5028c7a60333",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 97280,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0ac243f8385c32960e14__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_0ac243f8385c32960e14",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_0ac243f8385c32960e14",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e73321cec89d33336ea1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_e73321cec89d33336ea1",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_e73321cec89d33336ea1",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b7d9577ff884de98ea06__1080_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b7d9577ff884de98ea06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b7d9577ff884de98ea06",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1080,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_420788e663a2c3a810f6__5120_12288": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b7d9577ff884de98ea06__12288_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b7d9577ff884de98ea06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b7d9577ff884de98ea06",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_420788e663a2c3a810f6__40960_97280": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 40960,
            "at_most": 97280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_89505762fa1cbf33cb8f__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_89505762fa1cbf33cb8f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_89505762fa1cbf33cb8f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_f788ef57163552825d27": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_f788ef57163552825d27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f788ef57163552825d27",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
            "partial_rows": 2,
        },
    },
    "cake_rmsnorm_train_50044ee3bd1ca54bd821__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_50044ee3bd1ca54bd821",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_50044ee3bd1ca54bd821",
        "role": "bwd_small",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_ad2bb26ea5916baef963": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 6,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_ad2bb26ea5916baef963",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ad2bb26ea5916baef963",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_24643edc7e1143da24e0": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_24643edc7e1143da24e0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_24643edc7e1143da24e0",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_10239": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 10239,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_10239": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 10239,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3fd77c8ce97e8a523abb__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_3fd77c8ce97e8a523abb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3fd77c8ce97e8a523abb",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_16545056022785f2a7d8": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 6,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_16545056022785f2a7d8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_16545056022785f2a7d8",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_78c26a53e83c9803bcbf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_78c26a53e83c9803bcbf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_78c26a53e83c9803bcbf",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 f32 weight: (4096, 48640] the FP32-weight det pair the deterministic package ran at FlashInfer 2e989556 up to its edge T 48640 (main kernel 626aed74, dw reduce fcbfa14d with that package's host plan), renamed into this package, for speed (its dw is that package's): on GB300 it took 0.961x, 0.986x, 0.989x and 0.987x the time of the kblock single kernel 2911faf7, which FlashInfer main runs above T 9216, at T 10240, 16384, 32768 and 40960 (7 rounds, A/A within 0.9982 to 1.0024); the deterministic package now runs its row-issuer program below T 32768
    "cake_rmsnorm_train_626aed743358c27513f1__4096_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 f32 weight: (48640, 65536] the FP32-weight det pair the deterministic package ran above T 48640 at FlashInfer 2e989556 (main kernel cb164a27, the FP32-weight twin of 850eec3a, and dw reduce 5d75ea5f with that package's host plan), renamed into this package, for speed (its dw is that package's): on GB300 it took 0.976x and 0.980x the time of the kblock single kernel 2911faf7 at T 49152 and 65536 (7 rounds); the deterministic package now cuts it at T 51200
    "cake_rmsnorm_train_cb164a27c0157ed0712d__48640_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_cb164a27c0157ed0712d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cb164a27c0157ed0712d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 48640,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__48640_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 48640,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8e7f4d811e55a351a22e": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8e7f4d811e55a351a22e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8e7f4d811e55a351a22e",
        "role": "bwd_low",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_274b0055aace183f2367": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_274b0055aace183f2367",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_274b0055aace183f2367",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_cfaae3cec98f8501832b": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_cfaae3cec98f8501832b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cfaae3cec98f8501832b",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_122721bd6b4650866553__4096_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_122721bd6b4650866553": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56ee8e66479625fd078a__960_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_56ee8e66479625fd078a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56ee8e66479625fd078a",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 960,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_a546f410ec8cf03ef20d": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_a546f410ec8cf03ef20d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a546f410ec8cf03ef20d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_122721bd6b4650866553__960_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 960,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H5120 bf16 weight: (16, 64] the deterministic package's column-task program of these token counts (80 column tasks, ticket form), for speed: at T 64 its backward took 4.70 and 4.74 us against 6.00 and 5.98 us of the fixed-claim form it replaces and a forward and backward call 7.28 and 7.55 us against 8.40 and 8.64 us (GB300, FI bench, CUDA-graph span, L2 flushed before every call, two runs), and no sampled T from 24 to 64 had a backward call, a forward and backward call or a checkpointed step (two forward calls and a backward call) slower beyond the run-to-run band in all its runs (one or two); at T 96, 128, 256, 448 and from 640 to 1024 a forward and backward call or a checkpointed step was slower than with the det pair of the window above in both runs (8.80 and 8.70 us against 8.62 and 8.66 us for a forward and backward call at T 96), so that pair runs above T 64; the edge 16 against the window below is not measured yet
    "cake_rmsnorm_train_505283a5225ac45bc095__16_64": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_505283a5225ac45bc095",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_505283a5225ac45bc095",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__64_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__64_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__1024_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7ffb443a63a92ab6b0e0__5120_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_7ffb443a63a92ab6b0e0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7ffb443a63a92ab6b0e0",
        "role": "bwd_main",
        "select_rule": {
            "above": 5120,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 5120,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H5120 f32 weight: (16, 192] the deterministic package's column-task program of these token counts (80 column tasks, ticket form), for speed: at T 192 its backward took 6.94 and 6.88 us against 7.68 and 7.65 us of the det pair it replaces and a forward and backward call 9.71 and 9.76 us against 9.94 and 9.86 us (GB300, FI bench, CUDA-graph span, L2 flushed before every call, two runs), and no sampled T from 24 to 192 had a backward call, a forward and backward call or a checkpointed step (two forward calls and a backward call) slower beyond the run-to-run band in all its runs (one or two); FP64 dw error 0.73x the det pair's at T 64 (GB300, kernel-tests harness); at T 256 and from 640 to 1024 a forward and backward call and a checkpointed step were slower than with the det pair of the window above in both runs (10.68 and 10.64 us against 10.45 and 10.14 us for a forward and backward call at T 256), so that pair runs above T 192; the edge 16 against the window below is not measured yet
    "cake_rmsnorm_train_3a7bd162c370762719b3__16_192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a7bd162c370762719b3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_79811a76205d7d7dd197__192_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_79811a76205d7d7dd197",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_79811a76205d7d7dd197",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 192,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__192_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 192,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_79811a76205d7d7dd197__1024_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_79811a76205d7d7dd197",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_79811a76205d7d7dd197",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_f9f7d4cd75ef91cc8374__5120_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_f9f7d4cd75ef91cc8374",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f7d4cd75ef91cc8374",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 5120,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__1024_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H6144 bf16 weight: (16, 896] the deterministic package's column-task program of these token counts (96 column tasks, ticket form), for speed: at T 896 its backward took 13.97 us against 15.25 us of the det pair it replaces and a forward and backward call 19.58 us against 20.16 us (GB300, FI bench, CUDA-graph span, L2 flushed before every call), and no sampled T from 24 to 896 had a backward call, a forward and backward call or a checkpointed step (two forward calls and a backward call) slower beyond the run-to-run band in all its runs (one or two); FP64 dw error 0.75x the det pair's at T 64 (GB300, kernel-tests harness); at T 1024 its backward was still faster (14.94 us against 16.08 and 16.19 us in two runs) but a checkpointed step was slower in both runs (26.46 and 26.61 us against 25.95 and 25.87 us), so the det pair of the window above runs above T 896; the edge 16 against the window below is not measured yet
    "cake_rmsnorm_train_38b1c6bc8fa558bccc49__16_896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_38b1c6bc8fa558bccc49",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38b1c6bc8fa558bccc49",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__896_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "role": "bwd_small",
        "select_rule": {
            "above": 896,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__896_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 896,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "role": "bwd_small",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_322574af704b51aef0bb__4096_24575": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_322574af704b51aef0bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_322574af704b51aef0bb",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 24575,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_24575": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 24575,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H6144 f32 weight: (16, 1024] the deterministic package's column-task program of these token counts (96 column tasks, ticket form), for speed: at T 1024 its backward took 14.72 and 14.62 us against 16.10 and 16.19 us of the det pair it replaces and a forward and backward call 21.06 and 21.24 us against 21.63 and 21.26 us (GB300, FI bench, CUDA-graph span, L2 flushed before every call, two runs), and no sampled T from 24 to 1024 had a backward call, a forward and backward call or a checkpointed step (two forward calls and a backward call) slower beyond the run-to-run band in all its runs (one or two); above T 1024 the det pair of the window above runs, faster from T 1536 in an earlier FI bench run without the L2 flush (18.43 us against 19.31 us of this program); the edge 16 against the window below is not measured yet
    "cake_rmsnorm_train_017091c7827bdc8c95d5__16_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_017091c7827bdc8c95d5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_017091c7827bdc8c95d5",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_41b80b6af97974feb181__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41b80b6af97974feb181",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9a8c0396edd85d4d545e__4096_49152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_9a8c0396edd85d4d545e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a8c0396edd85d4d545e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H6144 residual bf16 weight: (56, 304] the deterministic package's column-task program of these token counts (96 column tasks; that package runs it from T 0 to 1216), in place of the det pair 0dcdc563 + 2e8d2128, a whole-SM program that program replaced in that package: FlashInfer's all-SM test excuses a det pair window that waits beside a kernel resident on every SM only where the deterministic package runs the same program; on GB300 it took 5.82, 8.13 and 9.34 us at T 64, 256 and 304 where the det pair took 7.65, 10.24 and 10.53 us, with 0.965 to 0.982 times the det pair's dw error against an FP64 reference (relative L2 or worst element, the larger over six input families)
    "cake_rmsnorm_train_07135550fae8b07bb82c__56_304": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_07135550fae8b07bb82c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07135550fae8b07bb82c",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 56,
            "at_most": 304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (48, 320] the FP32-weight twin 570eec50 of the deterministic package's column-task program 07135550 of these token counts (that package runs it from T 0 to 1216), in place of the det pair 56f988dd + 2e8d2128, a whole-SM program that program replaced in that package: FlashInfer's all-SM test excuses a det pair window that waits beside a kernel resident on every SM only where the deterministic package runs the same program; on GB300 it took 5.82, 8.13 and 9.34 us at T 64, 256 and 304 where the det pair took 7.65, 10.24 and 10.53 us, with 0.965 to 0.982 times the det pair's dw error against an FP64 reference (relative L2 or worst element, the larger over six input families) (BF16 weight; the FP32-weight twin computes the same dw)
    "cake_rmsnorm_train_570eec506ed94f36f062__48_320": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_570eec506ed94f36f062",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_570eec506ed94f36f062",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 48,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_7dc29629ff21625607f8__320_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_6a428da2227972ae18ae": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_6a428da2227972ae18ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6a428da2227972ae18ae",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_56d26996239835327501__16384_45056": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_56d26996239835327501",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56d26996239835327501",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1536 bf16 weight: (45056, 65536] the deterministic package's compensated det pair of these rows (main kernel e694b0a2: the register form of block-compensated in-chunk dw sums, a Kahan step every 4 steps; fixed-order dw reduce db5c0b67 with that package's host plan), renamed into this package, from T 45056, that package's accuracy edge, for speed and dw accuracy: the twosum single kernel dca25ace it replaces in (53248, 65536] took 73.54 us against 65.15 us for this pair in the deterministic package at T 65536 on R200 with 212 SMs (backward, harness pooled median), and the plain det pair 56d26996 + db5c0b67 it replaces in (45056, 53248] sums dw without compensation above that package's accuracy edge; the twosum single kernel keeps (65536, inf], where this pair exceeds the per-element budget at T 91148 (1.026) on R200 with 212 SMs; this pair's worst dw element over the per-element budget in this window is not sampled yet
    "cake_rmsnorm_train_e694b0a25ea01d1f6cdc__45056_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_e694b0a25ea01d1f6cdc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e694b0a25ea01d1f6cdc",
        "role": "bwd_main",
        "select_rule": {
            "above": 45056,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_db5c0b673c516bd0295d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_45056": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_db5c0b673c516bd0295d__45056_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 45056,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__320_1024_107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_45056": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_de8e07d4cb312b06dbc4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_de8e07d4cb312b06dbc4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1536 f32 weight: (45056, 65536] the deterministic package's compensated det pair of these rows (main kernel 0f0b782f, the FP32-weight twin of e694b0a2, and fixed-order dw reduce db5c0b67 with that package's host plan), renamed into this package, from T 45056, that package's accuracy edge, for dw accuracy: the single kernel 7c0dd63f it replaces in (53248, 65536] is the FP32-weight twin of 3f75f4a8 and computes its dw (FP32 dw sums over windows of 16 steps), which put its worst dw error at 1.18x the deterministic package's median FP64 error over 5 repeats at T 65536 on R200 with 212 SMs (BF16 weight, normal inputs), and the plain det pair de8e07d4 + db5c0b67 it replaces in (45056, 53248] sums dw without compensation above that package's accuracy edge; the twosum single kernel ccd94d87 keeps (65536, inf], where this pair exceeds the per-element budget at T 106874 (1.018) on R200 with 212 SMs; its time and its worst dw element over the per-element budget in this window are not measured yet
    "cake_rmsnorm_train_0f0b782f1e538caad3ce__45056_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_0f0b782f1e538caad3ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0f0b782f1e538caad3ce",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 45056,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_628248febbdfe17c3325__320_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_628248febbdfe17c3325",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_628248febbdfe17c3325",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_845cdcd8145ada631461__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_845cdcd8145ada631461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_845cdcd8145ada631461",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_81c3d8ab0a482e3af043": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_81c3d8ab0a482e3af043",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_81c3d8ab0a482e3af043",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_845cdcd8145ada631461__16384_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_845cdcd8145ada631461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_845cdcd8145ada631461",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_271082c56c222bf10d77": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_271082c56c222bf10d77",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_271082c56c222bf10d77",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c4d8be677fa70b841877": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_838a9a4054731f7942af__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_838a9a4054731f7942af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838a9a4054731f7942af",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_838a9a4054731f7942af__16384_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_838a9a4054731f7942af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838a9a4054731f7942af",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_a647fccc3d9fa9134733": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_a647fccc3d9fa9134733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a647fccc3d9fa9134733",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d41a514f5210e9adc39f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_d41a514f5210e9adc39f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d41a514f5210e9adc39f",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H4096 bf16 weight: (4096, 24064] the deterministic package's det pair of these rows (main kernel ee4a9053 and fixed-order dw reduce 10e24516 with that package's host plan), renamed into this package, up to T 24064, where that package switches to its compensated program: above T 24064 this pair put its worst dw error at 1.25x to 2.00x the deterministic package's median FP64 error over 5 repeats at T 32768 on R200 with 212 SMs (normal, cancel, concentrated, saturated_hi and saturated_lo inputs)
    "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_24064": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_ee4a9053bb25b8d49fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ee4a9053bb25b8d49fcf",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H4096 bf16 weight: (24064, 24576] the deterministic package's compensated det pair of these rows (main kernel 99a01f2a: block-compensated in-chunk dw sums over 8-step blocks; fixed-order dw reduce 10e24516 with that package's host plan), renamed into this package, for dw accuracy: the plain det pair ee4a9053 + 10e24516 it replaced in (24064, 36864] put its worst dw error at 1.25x to 2.00x the deterministic package's median FP64 error over 5 repeats at T 32768 on R200 with 212 SMs (normal, cancel, concentrated, saturated_hi and saturated_lo inputs), and the single kernel f11a8616 it replaced in (36864, 65536] (FlashInfer e14ddfd9 runs the same one there) put its worst dw error at 1.321x, 1.127x and 1.410x the median FP64 error of the deterministic package at FlashInfer 2e989556 over 5 repeats at T 40960, 49152 and 65536 on R200 (3 seeds; concentrated, saturated_hi and concentrated inputs), where this pair stays at 1.000x; the deterministic package at FlashInfer 2e989556 runs this pair at every T above 24064, the deterministic package now up to T 50176, above which its worst dw error passes 0.8 of the per-element budget on sm_107a; its time in this window is not measured yet; the pair-row single aed93ac9 runs above it, to T 65536, for speed
    "cake_rmsnorm_train_99a01f2a850874718c3f__24064_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_99a01f2a850874718c3f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_99a01f2a850874718c3f",
        "role": "bwd_main",
        "select_rule": {
            "above": 24064,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H4096 bf16 weight: (24576, 65536] the pair-row single kernel (32 items, one row per issue, two CTAs per SM) with its chunk table cut for 8 slots per SM (table_slots 8: more, shorter chunks than CTAs, the grid unchanged; twosum per-thread dw sums kept as FP32 pairs; each CTA writes its pairs as one row of a scratch area, and 256 helper CTAs fold the rows by pair addition and round dw once; no FP64 instruction) for speed: on R200 with 212 SMs (CUDA graph replay, cold L2, 7 rounds) it took 63.9 to 150 us at T 24577 to 65536, 0.96x to 0.98x the time of the deterministic package's compensated det pair 99a01f2a + 10e24516 it replaces (the single kernels FlashInfer e14ddfd9 runs there exceed 1.1x the deterministic package's dw error), with a dw error (relative L2 or worst element, the larger over six input families, 3 seeds x 5 repeats; dx equal to the deterministic package's) of at most 0.93x that of the deterministic package's program at FlashInfer 2e989556 (that det pair)
    "cake_rmsnorm_train_aed93ac9e4828f1b8b47__24576_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 8,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_aed93ac9e4828f1b8b47",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_aed93ac9e4828f1b8b47",
        "role": "bwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 32768,
        },
    },
    "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2030d80884deb8abcf7b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_2030d80884deb8abcf7b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2030d80884deb8abcf7b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_dc439f8335327e482025": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_dc439f8335327e482025",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dc439f8335327e482025",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b08633208e60b8ce94fd": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_b08633208e60b8ce94fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b08633208e60b8ce94fd",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0f1859b9fe9e47668a1c__4096_24064": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_0f1859b9fe9e47668a1c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0f1859b9fe9e47668a1c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_10e24516314c244771f4__4096_24064": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_10e24516314c244771f4__24064_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24064,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_10e24516314c244771f4__24064_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24064,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3feed5026e1156d2fba2__576_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "role": "bwd_main",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_dfaf378aa8ad93e68bbd": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3__16384_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H512 bf16 weight: (49152, inf] the twosum form of the H512 row instance (its knobs (4 reps, 4 slices) plus twosum and l2_ahead 2; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the det pair it replaces (the deterministic package runs the same pair there) exceeds the per-element budget at T 92912 (1.036, worst of 120 inputs of each of three families in both rstd forms) and at T 131072 (1.145) on R200 with 212 SMs, and its FP32-weight twin (the same dw computation) at T 80754 (1.012); the window keeps its start at T 49152 until a follow-up has qualifying evidence for the pair above it; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 49153 to 131072 (130 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.26x to 1.29x the pair's (the deterministic package's run of it) at T 49153, 65536, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_30501ccec48cc058333a__49152_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_30501ccec48cc058333a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_30501ccec48cc058333a",
        "role": "bwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
        },
    },
    "cake_rmsnorm_train_f94861c36a14eb75e39c__576_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d75459d901441384398b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_d75459d901441384398b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d75459d901441384398b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8f032213ba6eeb5d959f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b841d64bbcaefff5ce7b__384_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_b841d64bbcaefff5ce7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b841d64bbcaefff5ce7b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_383f709cd5c4fadd9436": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_383f709cd5c4fadd9436",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_383f709cd5c4fadd9436",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_03efa681eb38f59b00c2__16384_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_03efa681eb38f59b00c2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03efa681eb38f59b00c2",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H512 f32 weight: (49152, inf] the twosum form of the H512 row instance (its knobs (4 reps, 4 slices) plus twosum and l2_ahead 2; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the det pair it replaces (the deterministic package runs the same pair there) exceeds the per-element budget at T 80754 (1.012, worst of 120 inputs of each of three families in both rstd forms) and at T 118784 (1.097) on R200 with 212 SMs; the window keeps its start at T 49152 until a follow-up has qualifying evidence for the pair above it; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 49153 to 131072 (130 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 1.25x to 1.28x the pair's (the deterministic package's run of it) at T 49153, 65536, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_26fbe8e6f07a658e26f6__49152_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_26fbe8e6f07a658e26f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_26fbe8e6f07a658e26f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
        },
    },
    "cake_rmsnorm_train_f94861c36a14eb75e39c__384_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__256_511": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 511,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__640_895": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 640,
            "at_most": 895,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__896_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_6816fb80e97380f1bcbc": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_6816fb80e97380f1bcbc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6816fb80e97380f1bcbc",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b8b0913b40f035f43b76__19968_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_b8b0913b40f035f43b76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b8b0913b40f035f43b76",
        "role": "bwd_main",
        "select_rule": {
            "above": 19968,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__256_511": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 511,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__640_895": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 640,
            "at_most": 895,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__896_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19968,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_cdcd68d9227a76af4e88__19968_32767": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cdcd68d9227a76af4e88",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_cdcd68d9227a76af4e88",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19968,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H5120 f32 weight: (4096, 19968] the deterministic package's det pair of these rows (main kernel 468cc4d1, the FP32-weight twin of the plain program 6816fb80 the BF16-weight route runs here, and fixed-order dw reduce 2af4cf2f with that package's host plan), renamed into this package, for dw accuracy: the row single kernel dd795e84 it replaces put its worst dw error at 1.12x to 1.59x the deterministic package's median FP64 error over 5 repeats at T 14928 to 16384 on R200 with 212 SMs (normal, concentrated and saturated_lo inputs); the plain det pair's worst dw element stays within 0.545 of the per-element budget up to T 19968 on R200 with 212 SMs and within 0.499 with 216 SMs (60 or more inputs of each of three families, kernel rstd; both weight dtypes compute the same dw); its dw error exceeds the FP32 eager reference's at T 13024, 17472, 17728 and 19520 and at T 18304 (torch rstd) on R200 with 212 SMs
    "cake_rmsnorm_train_468cc4d1b8128187b4b3__4096_19968": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_468cc4d1b8128187b4b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_468cc4d1b8128187b4b3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_cd612de56c0985bb4caa__256_511": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 511,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_cd612de56c0985bb4caa__640_895": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 640,
            "at_most": 895,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_cd612de56c0985bb4caa__896_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H5120 f32 weight: (19968, 32767] the deterministic package's compensated det pair of these rows (main kernel de0cf457, the FP32-weight twin of 47da668e: block sums of the fold buffer over 8-step blocks; fixed-order TwoSum dw reduce cdcd68d9 with that package's host plan), renamed into this package, for dw accuracy: the compensated det pair 7b9e10b2 + 2af4cf2f (16-step blocks) it replaces put its worst dw error at 1.18x to 1.71x the deterministic package's median FP64 error over 5 repeats at T 32768 on R200 with 212 SMs (normal, cancel, concentrated, saturated_hi and saturated_lo inputs); this pair's worst dw element over the per-element budget is not sampled in this window yet; the pair-row single 9cdd6eef runs above, to T 131072
    "cake_rmsnorm_train_de0cf457cd240f9c6422__19968_32767": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_de0cf457cd240f9c6422",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_de0cf457cd240f9c6422",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19968,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d292172df772adb17ae__960_1151": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5d292172df772adb17ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5d292172df772adb17ae",
        "role": "bwd_main",
        "select_rule": {
            "above": 960,
            "at_most": 1151,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d292172df772adb17ae__1152_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5d292172df772adb17ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5d292172df772adb17ae",
        "role": "bwd_main",
        "select_rule": {
            "above": 1152,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H6144 bf16 weight: (19456, 30720] the det pair program; its worst dw error stays at or below 0.57 of the budget at every sampled T from 19457 to 30721 (every 64 rows) on R200, worst of 60 inputs of each of three families in both rstd forms; a follow-up returns this window to the pair of (3, 30720], which stays within the per-element budget at 66 sampled T from 19457 to 30720 (worst 0.913 at T 29680, worst of 120 inputs of each of three families in both rstd forms, R200 with 212 SMs) and takes 0.97x to 0.99x this det pair's backward time
    "cake_rmsnorm_train_19e18371f9243bd1a91a__19456_30720": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_19e18371f9243bd1a91a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_19e18371f9243bd1a91a",
        "role": "bwd_main",
        "select_rule": {
            "above": 19456,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__960_1151": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 960,
            "at_most": 1151,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__1152_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_30720": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19456,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H6144 f32 weight: (19456, 28672] the det pair program for dw accuracy: the pair of (20, 28672] exceeded the FP32 eager reference's dw error at 21 sampled T from 20416 rows on R200; its worst dw error stays within 0.631 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 19457 to 28673 (every 64 rows) on R200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_b83f8b3d546630a60675__19456_28672": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_b83f8b3d546630a60675",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b83f8b3d546630a60675",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19456,
            "at_most": 28672,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_28672": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19456,
            "at_most": 28672,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__320_447": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 320,
            "at_most": 447,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__3584_7168": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 3584,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_60bfcb24560bbae612c3__320_447": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 320,
            "at_most": 447,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 3584,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56f988dde54db37f09d9__3584_7168": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 3584,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    # sm_100a / sm_103a / sm_107a H256 bf16 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_780c64c6f2c4615bfbcf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 bf16 weight: (4, 24] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 24): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 24 are not measured yet
    "cake_rmsnorm_train_f2ccf7197a641c39acbf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_f2ccf7197a641c39acbf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f2ccf7197a641c39acbf",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 24,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 bf16 weight: (24, 64] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 64): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 24 and 64 are not measured yet
    "cake_rmsnorm_train_9fb6b410e91f9d92becf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_9fb6b410e91f9d92becf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9fb6b410e91f9d92becf",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 24,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 f32 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_67cb56412bd46304b711": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_67cb56412bd46304b711",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_67cb56412bd46304b711",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 f32 weight: (4, 24] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 24): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 24 are not measured yet
    "cake_rmsnorm_train_aa745395d7542d8a8e3a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_aa745395d7542d8a8e3a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_aa745395d7542d8a8e3a",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 24,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 f32 weight: (24, 64] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 64): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 24 and 64 are not measured yet
    "cake_rmsnorm_train_6a0114cb0b27a780a4c3": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 24,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 bf16 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_64a299691c17079fe510": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_64a299691c17079fe510",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64a299691c17079fe510",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 bf16 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_c167e0d26b363cf19e7b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_c167e0d26b363cf19e7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c167e0d26b363cf19e7b",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 f32 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_8c261dfc6e34351d7a11": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8c261dfc6e34351d7a11",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8c261dfc6e34351d7a11",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 f32 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_88e6252beaf7d056d555": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_88e6252beaf7d056d555",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_88e6252beaf7d056d555",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 bf16 weight: (0, 2] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 2 are not measured yet
    "cake_rmsnorm_train_db218c92c4c86b9da85a__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H1536 bf16 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_db218c92c4c86b9da85a__0_4": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (4, 5] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 5 are not measured yet
    "cake_rmsnorm_train_c2a523de1ff578621a88__4_5": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 5,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 bf16 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_c2a523de1ff578621a88__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (0, 2] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 2 are not measured yet
    "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H1536 f32 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_4": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (4, 6] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 6 are not measured yet
    "cake_rmsnorm_train_744aa77460c886f4f429__4_6": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_744aa77460c886f4f429",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_744aa77460c886f4f429",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 6,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 f32 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_744aa77460c886f4f429__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_744aa77460c886f4f429",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_744aa77460c886f4f429",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H2048 bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_c5f62407b828a85f82ec": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c5f62407b828a85f82ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5f62407b828a85f82ec",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 bf16 weight: (4, 12] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 12 are not measured yet
    "cake_rmsnorm_train_a951623c157d070e85ab__4_12": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a951623c157d070e85ab",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a951623c157d070e85ab",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H2048 bf16 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_a951623c157d070e85ab__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a951623c157d070e85ab",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a951623c157d070e85ab",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H2048 f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_0017ab41f68d8689ebc2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0017ab41f68d8689ebc2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0017ab41f68d8689ebc2",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (4, 14] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 14 are not measured yet
    "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_14": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 14,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H2048 f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H4096 bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_945a0afdb82ab47eb643": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_945a0afdb82ab47eb643",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_945a0afdb82ab47eb643",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (4, 12] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 12 are not measured yet
    "cake_rmsnorm_train_41f44b4bc0b765350a50__4_12": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H4096 bf16 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_41f44b4bc0b765350a50__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H4096 f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_f6bd1634a115bd281055": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f6bd1634a115bd281055",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f6bd1634a115bd281055",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (4, 14] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 14 are not measured yet
    "cake_rmsnorm_train_b378ba007691c582171e__4_14": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_b378ba007691c582171e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b378ba007691c582171e",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 14,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H4096 f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_b378ba007691c582171e__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_b378ba007691c582171e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b378ba007691c582171e",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H5120 bf16 weight: (0, 4] the one-cluster kernel (4 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_dbc680f356be4cd643de": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_dbc680f356be4cd643de",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dbc680f356be4cd643de",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (4, 12] the one-cluster kernel (4 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 12 are not measured yet
    "cake_rmsnorm_train_904b1ad45d6219b4f43c__4_12": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_904b1ad45d6219b4f43c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_904b1ad45d6219b4f43c",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H5120 f32 weight: (0, 4] the one-cluster kernel (4 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_d47138228b123e55fe5d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_d47138228b123e55fe5d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d47138228b123e55fe5d",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H5120 f32 weight: (4, 5] the one-cluster kernel (4 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 5 are not measured yet
    "cake_rmsnorm_train_05396cd3a077d3772fdc__4_5": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 5,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (4, 15] the one-cluster kernel (4 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 15 are not measured yet
    "cake_rmsnorm_train_05396cd3a077d3772fdc__4_15": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 15,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H6144 bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_e37752b056dd4e4a0bee": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e37752b056dd4e4a0bee",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e37752b056dd4e4a0bee",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (4, 6] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 6 are not measured yet
    "cake_rmsnorm_train_fdfb73350c1a926a4929__4_6": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 6,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_fdfb73350c1a926a4929__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H6144 f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 f32 weight: (4, 12] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 12 are not measured yet
    "cake_rmsnorm_train_18b8b8054c7053942bcd__4_12": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_18b8b8054c7053942bcd__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (0, 2] the one-cluster kernel (8 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 2 are not measured yet
    "cake_rmsnorm_train_deeb187630473818923d__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_deeb187630473818923d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_deeb187630473818923d",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H6144 residual bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_deeb187630473818923d__0_4": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_deeb187630473818923d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_deeb187630473818923d",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (4, 6] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 6 are not measured yet
    "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_6": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 4,
            "at_most": 6,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (4, 15] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 15 are not measured yet
    "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_15": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 4,
            "at_most": 15,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H6144 residual f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 are not measured yet
    "cake_rmsnorm_train_a6b1a6e565dc98b49e1a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
        "role": "bwd_tiny_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (4, 10] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 10 are not measured yet
    "cake_rmsnorm_train_c29a8f856283578103dc__4_10": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c29a8f856283578103dc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c29a8f856283578103dc",
        "role": "bwd_tiny_w32_residual",
        "select_rule": {
            "above": 4,
            "at_most": 10,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order dw exact to second order, the same bits on every launch; its time against the program it replaces and the dx class at the edge 4 and 16 are not measured yet
    "cake_rmsnorm_train_c29a8f856283578103dc__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c29a8f856283578103dc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c29a8f856283578103dc",
        "role": "bwd_tiny_w32_residual",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_train_21d3416baa877808371e",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41__low",
            "cake_rmsnorm_train_1974c5c86cee63f33370",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b44107e05796c2922187",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4__cut",
            "cake_rmsnorm_train_b44107e05796c2922187__band",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_train_0da8e18d4af432c6927a",
            "cake_rmsnorm_train_a4d3f82b3e776f4fc49b__cut",
            "cake_rmsnorm_train_0da8e18d4af432c6927a__band",
            "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        ],
        "fwd_h256": [
            "cake_rmsnorm_train_0489c784c13bb25499f6",
            "cake_rmsnorm_train_1b284baec9663d6de1b9",
            "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
            "cake_rmsnorm_train_1388ab779a35c4e41efb",
            "cake_rmsnorm_train_466331ff3cd2d6c62738",
            "cake_rmsnorm_train_ed9483d8c356affee55d",
            "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_train_f2e721db4c30900f12fe",
            "cake_rmsnorm_train_77456435c9fbd1d18fcf",
            "cake_rmsnorm_train_42d82d26e97c36651dae",
            "cake_rmsnorm_train_b0299f6e0881fb35f27b",
            "cake_rmsnorm_train_32bebe5a7782311d60a0",
            "cake_rmsnorm_train_d0a656f053396a58c49a",
            "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_train_4371270e4db4b985998f",
            "cake_rmsnorm_train_f0db3068d0d05442a59d__low",
            "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid",
            "cake_rmsnorm_train_f0db3068d0d05442a59d",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_8757f6da86abf54363d5",
            "cake_rmsnorm_train_7673978dc698790054d2",
            "cake_rmsnorm_train_3d20ab4a432bddf7349a__100",
            "cake_rmsnorm_train_3430e5d9c053309b0323__100",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_train_6769e393999bc2ba1926",
            "cake_rmsnorm_train_e2eabcb3b0915d916a13",
            "cake_rmsnorm_train_7da36ff557bb2bac8d89__100",
            "cake_rmsnorm_train_3c1e1f571f15d42e8d57__100",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__small",
            "cake_rmsnorm_train_b5e173f5ecc736d91109__mid",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__ov100_0_640",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_640_6144",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__ov100_6144_16384",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_16384_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace__100",
            "cake_rmsnorm_train_bad4db9153cc903b7ace__ps5",
            "cake_rmsnorm_train_288d7b1d51c340236cb7",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__100",
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ps5",
            "cake_rmsnorm_train_3d619cd1e3ae638cce3d__ov100_1024_6144",
            "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov100_6144_inf",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_308da18230017dadc02d",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_train_8e3407e059cfc005f421",
            "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_train_6325c474ad9955a319e9",
        ],
        "bwd_h256": [
            "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
            "cake_rmsnorm_train_f2ccf7197a641c39acbf",
            "cake_rmsnorm_train_9fb6b410e91f9d92becf",
            "cake_rmsnorm_train_195c4317b51bb4ea1137",
            "cake_rmsnorm_train_172a47a15d4c1a76c061__20480_92160",
            "cake_rmsnorm_train_083ea823c691e0631ad3__92160_524288",
            "cake_rmsnorm_train_172a47a15d4c1a76c061",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__92160_524288",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_67cb56412bd46304b711",
            "cake_rmsnorm_train_aa745395d7542d8a8e3a",
            "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
            "cake_rmsnorm_train_8de0131445eed6eeabb7",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__20480_92160",
            "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__92160_864240",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__864240_inf",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__92160_864240",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_64a299691c17079fe510",
            "cake_rmsnorm_train_c167e0d26b363cf19e7b",
            "cake_rmsnorm_train_5f3e3fccaea795d0a806__16_227",
            "cake_rmsnorm_train_09601d42628de24f6f87__227_1024",
            "cake_rmsnorm_train_123db0b2f7fb38cfab03__1024_1280",
            "cake_rmsnorm_train_d0bde38aa443d42a5f04",
            "cake_rmsnorm_train_ad1a5a3c5f88e3d10036__4096_32768",
            "cake_rmsnorm_train_8877322a3002e3a4fefd__32768_inf",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_train_fef03ce50a792c8704bd__1024_1280",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553__32768_inf",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_8c261dfc6e34351d7a11",
            "cake_rmsnorm_train_88e6252beaf7d056d555",
            "cake_rmsnorm_train_942503b80a21f22cd392__16_227",
            "cake_rmsnorm_train_aa7c5a5407cba063b9d5__227_292",
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__292_960",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__292_960",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_db218c92c4c86b9da85a__0_2",
            "cake_rmsnorm_train_38e972a94141a1961c90__2_64",
            "cake_rmsnorm_train_dc478ac1195882aedee6__64_384",
            "cake_rmsnorm_train_7dc29629ff21625607f8__384_1024",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_21c8bb55c8d672b63952",
            "cake_rmsnorm_train_b457c4d134e44476dc6f__32768_36352",
            "cake_rmsnorm_train_b5bdabba0b80fb8afcc5__36352_78784",
            "cake_rmsnorm_train_66238d219de588fca483",
            "cake_rmsnorm_train_b42efd6bf06148d75151__131072_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_36352",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__36352_78784",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_2",
            "cake_rmsnorm_train_d5a621af7a24e403b5f3__2_64",
            "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e__64_384",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__384_1024",
            "cake_rmsnorm_train_4b6981c10fd40a9d8aec__1024_4096",
            "cake_rmsnorm_train_4bf450b6af0314ed03be",
            "cake_rmsnorm_train_4b6981c10fd40a9d8aec__32768_75776",
            "cake_rmsnorm_train_5e78d78f2d8203948198",
            "cake_rmsnorm_train_e2b99fee32764cfd1295__131072_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_c5f62407b828a85f82ec",
            "cake_rmsnorm_train_a951623c157d070e85ab__4_12",
            "cake_rmsnorm_train_ef052c0918725ffa68d4__12_64",
            "cake_rmsnorm_train_64e6aa82bb2c6d4ea978__64_336",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__336_3072",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__4096_7168",
            "cake_rmsnorm_train_0fa4f255caf709d09f66",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_16384",
            "cake_rmsnorm_train_ceeef043ac64f37fc529__16384_58880",
            "cake_rmsnorm_train_8b8cf4f5b82e77554908",
            "cake_rmsnorm_train_83f905ffd00c0ef641e2__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_7168",
            "cake_rmsnorm_train_d0960be986e58f449f50__11264_16384",
            "cake_rmsnorm_train_d0960be986e58f449f50__16384_58880",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_0017ab41f68d8689ebc2",
            "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_14",
            "cake_rmsnorm_train_382e45788ace3ff40ba1__14_64",
            "cake_rmsnorm_train_17493054fed5409b6df9__64_336",
            "cake_rmsnorm_train_17493054fed5409b6df9__336_1024",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__1024_4096",
            "cake_rmsnorm_train_b187e5989302def22b2d__4096_15403",
            "cake_rmsnorm_train_ba7dbf6a07744268705e",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__45056_58880",
            "cake_rmsnorm_train_25068edea92269e1bdb1__58880_65536",
            "cake_rmsnorm_train_273d3f9b357669a821e2",
            "cake_rmsnorm_train_9f775c72c1335f4b08ac__131072_inf",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_15403",
            "cake_rmsnorm_train_d0960be986e58f449f50__58880_65536",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_945a0afdb82ab47eb643",
            "cake_rmsnorm_train_41f44b4bc0b765350a50__4_12",
            "cake_rmsnorm_train_231e01eca0459c2c0bf5__12_64",
            "cake_rmsnorm_train_4822793ea0c954c11af6__64_1152",
            "cake_rmsnorm_train_9a830956447d4e304914__1152_4096",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_4608",
            "cake_rmsnorm_train_2443b1f0183048c47b15__4608_16384",
            "cake_rmsnorm_train_03f6488b89b5f9ae3db5__16384_36864",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_44032",
            "cake_rmsnorm_train_7122dbcd697b3fbe1ee0__44032_65536",
            "cake_rmsnorm_train_fd7ae1d7547fca6156a0",
            "cake_rmsnorm_train_c67fc8b9d1b1edb8d7da__131072_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1152_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_4608",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__44032_65536",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_f6bd1634a115bd281055",
            "cake_rmsnorm_train_b378ba007691c582171e__4_14",
            "cake_rmsnorm_train_ce1f7d216ecc0a1573b9__14_64",
            "cake_rmsnorm_train_8616070567821eaa3e35__64_1792",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec__1792_3584",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec",
            "cake_rmsnorm_train_626aed743358c27513f1__36864_44032",
            "cake_rmsnorm_train_5ef5f1f13ed8f2b994c4__44032_65536",
            "cake_rmsnorm_train_6e65495154249f51e74b",
            "cake_rmsnorm_train_9ebd0232954f1b1842f7__131072_inf",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__44032_65536",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_dbc680f356be4cd643de",
            "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__4_64",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__64_896",
            "cake_rmsnorm_train_b482c84dbffcd75a4a4c__896_6144",
            "cake_rmsnorm_train_c145ef195d7aeb04f242__6144_56256",
            "cake_rmsnorm_train_56634f53e6456b34f76f__56256_131072",
            "cake_rmsnorm_train_ade9f6a4f10b069ba8a3__131072_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__896_6144",
            "cake_rmsnorm_train_c5831b29d17867842e5f__6144_56256",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_d47138228b123e55fe5d",
            "cake_rmsnorm_train_244b7d841d62d9721c99__4_64",
            "cake_rmsnorm_train_68f6e584c687c30c0084__64_1792",
            "cake_rmsnorm_train_de8f6091077dc7ac5947__1792_6144",
            "cake_rmsnorm_train_9479b1298a1b19d4d996__6144_56256",
            "cake_rmsnorm_train_6cedccdf3a58f23f260f__56256_131072",
            "cake_rmsnorm_train_fc2a52a2c9bf8c5be9af__131072_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__6144_56256",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_e37752b056dd4e4a0bee",
            "cake_rmsnorm_train_d70f2562a592151ec4f9__4_64",
            "cake_rmsnorm_train_44508df7dab2ab688f85__64_1152",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096",
            "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_42496",
            "cake_rmsnorm_train_9433061da18b2db4299d__42496_65536",
            "cake_rmsnorm_train_995b2229fe43c772b2c7",
            "cake_rmsnorm_train_10a41132a42d594ce6dd__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_42496",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_deeb187630473818923d__0_2",
            "cake_rmsnorm_train_7b0f5a6a8769473c322e__2_64",
            "cake_rmsnorm_train_864e84f3d874f850c091__64_1408",
            "cake_rmsnorm_train_eb332e8c97f6387027c6__1408_65536",
            "cake_rmsnorm_train_7c216db7f9e0c6fd9c4d",
            "cake_rmsnorm_train_1609f7969c50fba7f2c7__131072_inf",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
            "cake_rmsnorm_train_9b37923697ee8ca52a9e__4_64",
            "cake_rmsnorm_train_b3e6531e46667901ab91__64_1168",
            "cake_rmsnorm_train_41b80b6af97974feb181__1168_3584",
            "cake_rmsnorm_train_a562656cef5bf5dda160__3584_65536",
            "cake_rmsnorm_train_d5a17016b7ff2c6e3ba9",
            "cake_rmsnorm_train_45cf60e3cd8603c2abeb__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1168_3584",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
            "cake_rmsnorm_train_ac9d69f94aabfc68aa40__4_64",
            "cake_rmsnorm_train_71d36617003ab7742a77__64_1408",
            "cake_rmsnorm_train_a640d0b799889f785ad5__1408_65536",
            "cake_rmsnorm_train_0bdcd2c63a9a4c5cb9d4",
            "cake_rmsnorm_train_d3d674373d17aaf425c0__131072_inf",
        ],
    },
    "sm_103a": {
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_train_21d3416baa877808371e",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41__low",
            "cake_rmsnorm_train_1974c5c86cee63f33370",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_c5a8099e177e463daab9",
            "cake_rmsnorm_train_b44107e05796c2922187__mid",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_train_01f0ebf3db9fa2c0884e",
            "cake_rmsnorm_train_0da8e18d4af432c6927a__mid",
            "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        ],
        "fwd_h256": [
            "cake_rmsnorm_train_0489c784c13bb25499f6",
            "cake_rmsnorm_train_bbb05314930d192dfe7b",
            "cake_rmsnorm_train_1b284baec9663d6de1b9__20480_40960",
            "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
            "cake_rmsnorm_train_466331ff3cd2d6c62738__81920_114688",
            "cake_rmsnorm_train_86d51eff64fc9fe57cf0__114688_196608",
            "cake_rmsnorm_train_ed9483d8c356affee55d",
            "cake_rmsnorm_train_37f663183cbff23c2893",
            "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5__368640_inf",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_train_f2e721db4c30900f12fe",
            "cake_rmsnorm_train_a8fffd93682c5e382dc5",
            "cake_rmsnorm_train_77456435c9fbd1d18fcf__20480_40960",
            "cake_rmsnorm_train_42d82d26e97c36651dae",
            "cake_rmsnorm_train_32bebe5a7782311d60a0__81920_114688",
            "cake_rmsnorm_train_42d82d26e97c36651dae__114688_196608",
            "cake_rmsnorm_train_d0a656f053396a58c49a",
            "cake_rmsnorm_train_18e3c0a5204006b3a56f",
            "cake_rmsnorm_train_670bfb3eed0ddf6ba461__368640_inf",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_train_4371270e4db4b985998f",
            "cake_rmsnorm_train_f0db3068d0d05442a59d__low",
            "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid103",
            "cake_rmsnorm_train_bdc524a233d2bc6e9571__wide103",
            "cake_rmsnorm_train_f0db3068d0d05442a59d",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_8757f6da86abf54363d5",
            "cake_rmsnorm_train_7673978dc698790054d2",
            "cake_rmsnorm_train_3d20ab4a432bddf7349a",
            "cake_rmsnorm_train_3430e5d9c053309b0323",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_train_6769e393999bc2ba1926",
            "cake_rmsnorm_train_e2eabcb3b0915d916a13",
            "cake_rmsnorm_train_7da36ff557bb2bac8d89",
            "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__small",
            "cake_rmsnorm_train_b5e173f5ecc736d91109__mid",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_train_1defffdf2712cae36784__ov103_0_640",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_640_6144",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_6144_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_train_288d7b1d51c340236cb7",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ov103_1024_6144",
            "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov103_6144_inf",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_308da18230017dadc02d",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_train_8e3407e059cfc005f421",
            "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_train_6325c474ad9955a319e9",
        ],
        "bwd_h256": [
            "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
            "cake_rmsnorm_train_f2ccf7197a641c39acbf",
            "cake_rmsnorm_train_9fb6b410e91f9d92becf",
            "cake_rmsnorm_train_083ea823c691e0631ad3__64_1080",
            "cake_rmsnorm_train_af005462cd6a301f31c4__1080_5120",
            "cake_rmsnorm_train_a385ae1681a2c078344e__5120_12288",
            "cake_rmsnorm_train_af005462cd6a301f31c4__12288_40960",
            "cake_rmsnorm_train_a385ae1681a2c078344e__40960_97280",
            "cake_rmsnorm_train_083ea823c691e0631ad3__97280_720896",
            "cake_rmsnorm_train_0291949ba535c646c3a8__720896_1572864",
            "cake_rmsnorm_train_c18297d75c0d424978fd",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__64_1080",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_5120",
            "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__40960_97280",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__97280_720896",
            "cake_rmsnorm_train_0ac243f8385c32960e14__720896_1572864",
            "cake_rmsnorm_train_e73321cec89d33336ea1",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_67cb56412bd46304b711",
            "cake_rmsnorm_train_aa745395d7542d8a8e3a",
            "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
            "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__64_1080",
            "cake_rmsnorm_train_b7d9577ff884de98ea06__1080_5120",
            "cake_rmsnorm_train_420788e663a2c3a810f6__5120_12288",
            "cake_rmsnorm_train_b7d9577ff884de98ea06__12288_40960",
            "cake_rmsnorm_train_420788e663a2c3a810f6__40960_97280",
            "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__97280_720896",
            "cake_rmsnorm_train_89505762fa1cbf33cb8f__720896_1572864",
            "cake_rmsnorm_train_f788ef57163552825d27",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__64_1080",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_5120",
            "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__40960_97280",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__97280_720896",
            "cake_rmsnorm_train_0ac243f8385c32960e14__720896_1572864",
            "cake_rmsnorm_train_e73321cec89d33336ea1",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_64a299691c17079fe510",
            "cake_rmsnorm_train_c167e0d26b363cf19e7b",
            "cake_rmsnorm_train_877751ba915bbc261064__16_64",
            "cake_rmsnorm_train_09601d42628de24f6f87__64_1024",
            "cake_rmsnorm_train_8e7f4d811e55a351a22e",
            "cake_rmsnorm_train_274b0055aace183f2367",
            "cake_rmsnorm_train_cfaae3cec98f8501832b",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__64_1024",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_train_122721bd6b4650866553__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_8c261dfc6e34351d7a11",
            "cake_rmsnorm_train_88e6252beaf7d056d555",
            "cake_rmsnorm_train_81d9adbf31a26a4ebc02__16_227",
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__227_960",
            "cake_rmsnorm_train_56ee8e66479625fd078a__960_32768",
            "cake_rmsnorm_train_a546f410ec8cf03ef20d",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_960",
            "cake_rmsnorm_train_122721bd6b4650866553__960_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_db218c92c4c86b9da85a__0_4",
            "cake_rmsnorm_train_c2a523de1ff578621a88__4_5",
            "cake_rmsnorm_train_38e972a94141a1961c90__0_64_103",
            "cake_rmsnorm_train_53285b9598170177d3e0__64_214",
            "cake_rmsnorm_train_7dc29629ff21625607f8__214_1024",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_1df7db17277d363b5168",
            "cake_rmsnorm_train_4df08ce51fb01ba1193f",
            "cake_rmsnorm_train_f522f5b8d4343d96ca04__15566_46592",
            "cake_rmsnorm_train_1df502fd0e17091869b7__46592_53184",
            "cake_rmsnorm_train_cbeae168d70edd196417__53184_76352",
            "cake_rmsnorm_train_1df502fd0e17091869b7__76352_131072",
            "cake_rmsnorm_train_327bfeb4ab8fefc2731d__131072_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__214_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_train_1863320f3158d35eead6__15566_46592",
            "cake_rmsnorm_train_882ec880717f382e43eb__53184_76352",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_4",
            "cake_rmsnorm_train_744aa77460c886f4f429__4_6",
            "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64_103",
            "cake_rmsnorm_train_5c5be9a6aeac20fd2776__64_214",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__214_1024",
            "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
            "cake_rmsnorm_train_a99e2b380514c4d86d60",
            "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2",
            "cake_rmsnorm_train_109f19bb177c8f20cadc__15566_37888",
            "cake_rmsnorm_train_9a47ed21f63c848f2a59__37888_40448",
            "cake_rmsnorm_train_ca6211f3ea2ce8c86a05__40448_76352",
            "cake_rmsnorm_train_9a47ed21f63c848f2a59__76352_131072",
            "cake_rmsnorm_train_963a7ca8d03f78bbe6f9__131072_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__214_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_train_1863320f3158d35eead6__15566_37888",
            "cake_rmsnorm_train_882ec880717f382e43eb__40448_76352",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_c5f62407b828a85f82ec",
            "cake_rmsnorm_train_a951623c157d070e85ab__4_16",
            "cake_rmsnorm_train_ef052c0918725ffa68d4__16_63",
            "cake_rmsnorm_train_2b4c0f5b0e9c312fa43e__63_171",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__171_3072",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__large",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768",
            "cake_rmsnorm_train_302f64db6abe79e5259e__32768_58496",
            "cake_rmsnorm_train_018f0cd723106f709f11__58496_61952",
            "cake_rmsnorm_train_018f0cd723106f709f11__61952_68160",
            "cake_rmsnorm_train_2c32330b326a6db3f77b__68160_131072",
            "cake_rmsnorm_train_8bd89a4c834f80db8a34__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__171_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496",
            "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952",
            "cake_rmsnorm_train_d0960be986e58f449f50__61952_68160",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_0017ab41f68d8689ebc2",
            "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_16",
            "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64_103",
            "cake_rmsnorm_train_2e9171b7753650cda515__64_171",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__171_3072",
            "cake_rmsnorm_train_d96a14fe8882bbd9ec2e",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__large",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__16384_32768",
            "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_58496",
            "cake_rmsnorm_train_ca8ca4674a3dbd89ad16__58496_61952",
            "cake_rmsnorm_train_ca8ca4674a3dbd89ad16__61952_68160",
            "cake_rmsnorm_train_76e43a7d9c7ea1f59e0d__68160_131072",
            "cake_rmsnorm_train_88c6c6b2af1922f4c1a9__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__171_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496",
            "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952",
            "cake_rmsnorm_train_d0960be986e58f449f50__61952_68160",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_945a0afdb82ab47eb643",
            "cake_rmsnorm_train_41f44b4bc0b765350a50__4_16",
            "cake_rmsnorm_train_3dbab2323fa42435a72d__16_32",
            "cake_rmsnorm_train_ff0156d59b2b99144701__32_152",
            "cake_rmsnorm_train_50044ee3bd1ca54bd821__152_256",
            "cake_rmsnorm_train_ad2bb26ea5916baef963",
            "cake_rmsnorm_train_24643edc7e1143da24e0",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_10239",
            "cake_rmsnorm_train_06e5f7e09e42d1108365__10239_65536",
            "cake_rmsnorm_train_3e6231b66942e510fdae__65536_262144",
            "cake_rmsnorm_train_8c8bebecc5b08454b6a0__262144_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__152_256",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_10239",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_f6bd1634a115bd281055",
            "cake_rmsnorm_train_b378ba007691c582171e__4_16",
            "cake_rmsnorm_train_c1b9ee6b9f10766cd537__16_32",
            "cake_rmsnorm_train_11820fe48711775d5685__32_152",
            "cake_rmsnorm_train_3fd77c8ce97e8a523abb__152_256",
            "cake_rmsnorm_train_16545056022785f2a7d8",
            "cake_rmsnorm_train_78c26a53e83c9803bcbf",
            "cake_rmsnorm_train_626aed743358c27513f1__4096_48640",
            "cake_rmsnorm_train_cb164a27c0157ed0712d__48640_65536",
            "cake_rmsnorm_train_2911faf7a786144495a6__65536_262144",
            "cake_rmsnorm_train_131a495c5897206b6874__262144_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__152_256",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_48640",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__48640_65536",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_dbc680f356be4cd643de",
            "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__4_16",
            "cake_rmsnorm_train_505283a5225ac45bc095__16_64",
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__64_1024",
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__1024_5120",
            "cake_rmsnorm_train_7ffb443a63a92ab6b0e0__5120_61952",
            "cake_rmsnorm_train_b523a764b029e5ca1aaf__61952_393216",
            "cake_rmsnorm_train_9e4c0825df92af02074f__393216_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__64_1024",
            "cake_rmsnorm_train_c5831b29d17867842e5f__1024_5120",
            "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_d47138228b123e55fe5d",
            "cake_rmsnorm_train_05396cd3a077d3772fdc__4_5",
            "cake_rmsnorm_train_244b7d841d62d9721c99__5_16",
            "cake_rmsnorm_train_3a7bd162c370762719b3__16_192",
            "cake_rmsnorm_train_79811a76205d7d7dd197__192_1024",
            "cake_rmsnorm_train_79811a76205d7d7dd197__1024_5120",
            "cake_rmsnorm_train_f9f7d4cd75ef91cc8374__5120_61952",
            "cake_rmsnorm_train_90598c14b4e396f5c8b0__61952_196608",
            "cake_rmsnorm_train_f165933caadda3434828__196608_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__192_1024",
            "cake_rmsnorm_train_c5831b29d17867842e5f__1024_5120",
            "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_e37752b056dd4e4a0bee",
            "cake_rmsnorm_train_fdfb73350c1a926a4929__4_6",
            "cake_rmsnorm_train_d70f2562a592151ec4f9__6_16",
            "cake_rmsnorm_train_38b1c6bc8fa558bccc49__16_896",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__896_1024",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1024_4096",
            "cake_rmsnorm_train_322574af704b51aef0bb__4096_24575",
            "cake_rmsnorm_train_f5806bd223b2ad27230a__24575_65536",
            "cake_rmsnorm_train_7236b7379272c82c7614__65536_131072",
            "cake_rmsnorm_train_582b2a20f9be648af8ec__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__896_1024",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1024_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_24575",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_deeb187630473818923d__0_4",
            "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_6",
            "cake_rmsnorm_train_7b0f5a6a8769473c322e__6_31",
            "cake_rmsnorm_train_07135550fae8b07bb82c__31_32",
            "cake_rmsnorm_train_7b0f5a6a8769473c322e__32_48",
            "cake_rmsnorm_train_864e84f3d874f850c091__48_56",
            "cake_rmsnorm_train_07135550fae8b07bb82c__56_304",
            "cake_rmsnorm_train_864e84f3d874f850c091__304_1920",
            "cake_rmsnorm_train_601fb35c90bc116731b1__1920_131072",
            "cake_rmsnorm_train_52831494c1b62db758fd",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
            "cake_rmsnorm_train_18b8b8054c7053942bcd__4_12",
            "cake_rmsnorm_train_9b37923697ee8ca52a9e__12_16",
            "cake_rmsnorm_train_017091c7827bdc8c95d5__16_1024",
            "cake_rmsnorm_train_41b80b6af97974feb181__1024_4096",
            "cake_rmsnorm_train_9a8c0396edd85d4d545e__4096_49152",
            "cake_rmsnorm_train_aa46aa07d86900ad832f__49152_83392",
            "cake_rmsnorm_train_935b8bea9166f29912ff__83392_131072",
            "cake_rmsnorm_train_4f9a76e68671f213ab7c__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1024_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__49152_83392",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
            "cake_rmsnorm_train_c29a8f856283578103dc__4_10",
            "cake_rmsnorm_train_ac9d69f94aabfc68aa40__10_31",
            "cake_rmsnorm_train_570eec506ed94f36f062__31_32",
            "cake_rmsnorm_train_ac9d69f94aabfc68aa40__32_48",
            "cake_rmsnorm_train_570eec506ed94f36f062__48_320",
            "cake_rmsnorm_train_71d36617003ab7742a77__320_1824",
            "cake_rmsnorm_train_cf1cf64436893a2a2d03__1824_131072",
            "cake_rmsnorm_train_56c46a201c168050aa60",
        ],
    },
    "sm_107a": {
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f__107",
            "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
            "cake_rmsnorm_train_682675bda578f5b7a312",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_train_21d3416baa877808371e__107",
            "cake_rmsnorm_train_92cf7b87f1ebffb1b12b",
            "cake_rmsnorm_train_a30e5a3b69b7c183992f",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b44107e05796c2922187__small",
            "cake_rmsnorm_train_a4da85136fba4588239f",
            "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
            "cake_rmsnorm_train_0ad4d53c4a0df501a99d__wide",
            "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_train_0da8e18d4af432c6927a__small",
            "cake_rmsnorm_train_a72c6ece1900f748eaa2",
            "cake_rmsnorm_train_595f506004038a65cc63",
            "cake_rmsnorm_train_40210950f2e7ee48bc3e__ov107_12288_16384",
            "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        ],
        "fwd_h256": [
            "cake_rmsnorm_train_c5ca409f44e169148301",
            "cake_rmsnorm_train_bbb05314930d192dfe7b__6144_32768",
            "cake_rmsnorm_train_466331ff3cd2d6c62738__32768_98304",
            "cake_rmsnorm_train_37f663183cbff23c2893__98304_inf",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_train_8f2756937eaac6b02502",
            "cake_rmsnorm_train_a8fffd93682c5e382dc5__6144_32768",
            "cake_rmsnorm_train_32bebe5a7782311d60a0__32768_98304",
            "cake_rmsnorm_train_18e3c0a5204006b3a56f__98304_inf",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba__107",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low107",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid107",
            "cake_rmsnorm_train_bf933e1660cab8e5787b",
            "cake_rmsnorm_train_bf933e1660cab8e5787b__wide",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__107",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_train_4371270e4db4b985998f__107",
            "cake_rmsnorm_train_f0db3068d0d05442a59d__low107",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_1024_4096",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_4096_20480",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_20480_49152",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_49152_inf",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_7673978dc698790054d2__quad",
            "cake_rmsnorm_train_2a42e5c02b240f8709b1",
            "cake_rmsnorm_train_60f47c5d00570ecf7740",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_train_e2eabcb3b0915d916a13__quad",
            "cake_rmsnorm_train_79fecc7016e44e940733",
            "cake_rmsnorm_train_61accbcc02891615f286",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_train_fd1097c45c1e328871fd__107",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__low107",
            "cake_rmsnorm_train_b5e173f5ecc736d91109__mid107",
            "cake_rmsnorm_train_044d4a2aa0553c16db06__large107",
            "cake_rmsnorm_train_044d4a2aa0553c16db06__wide107",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__107",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__107",
            "cake_rmsnorm_train_1defffdf2712cae36784__low107",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__mid107",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__large107",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__wide107",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__ov107_49152_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
            "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
            "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
            "cake_rmsnorm_train_fdfa500c83d95df0f5c1",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_e646bbdbd852884953da__small",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_train_6325c474ad9955a319e9__small",
            "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_train_6325c474ad9955a319e9",
        ],
        "bwd_h256": [
            "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
            "cake_rmsnorm_train_f2ccf7197a641c39acbf",
            "cake_rmsnorm_train_9fb6b410e91f9d92becf",
            "cake_rmsnorm_train_083ea823c691e0631ad3__64_786432",
            "cake_rmsnorm_train_172a47a15d4c1a76c061__786432_inf",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__64_786432",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_67cb56412bd46304b711",
            "cake_rmsnorm_train_aa745395d7542d8a8e3a",
            "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
            "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__64_786432",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__786432_inf",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__64_786432",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_64a299691c17079fe510",
            "cake_rmsnorm_train_c167e0d26b363cf19e7b",
            "cake_rmsnorm_train_bcd8d6a9c739c7a11c36__16_576",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__576_1024",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096",
            "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
            "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3__16384_49152",
            "cake_rmsnorm_train_30501ccec48cc058333a__49152_inf",
            "cake_rmsnorm_train_f94861c36a14eb75e39c__576_1024",
            "cake_rmsnorm_train_d75459d901441384398b",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_8c261dfc6e34351d7a11",
            "cake_rmsnorm_train_88e6252beaf7d056d555",
            "cake_rmsnorm_train_685b53bf6ce1b6ff13f8__16_384",
            "cake_rmsnorm_train_b841d64bbcaefff5ce7b__384_1024",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22__1024_4096",
            "cake_rmsnorm_train_383f709cd5c4fadd9436",
            "cake_rmsnorm_train_03efa681eb38f59b00c2__16384_49152",
            "cake_rmsnorm_train_26fbe8e6f07a658e26f6__49152_inf",
            "cake_rmsnorm_train_f94861c36a14eb75e39c__384_1024",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_db218c92c4c86b9da85a__0_4",
            "cake_rmsnorm_train_c2a523de1ff578621a88__4_16",
            "cake_rmsnorm_train_9e71ed0d9ad7a7739002__16_320",
            "cake_rmsnorm_train_7dc29629ff21625607f8__320_1024",
            "cake_rmsnorm_train_6a428da2227972ae18ae",
            "cake_rmsnorm_train_56d26996239835327501__16384_45056",
            "cake_rmsnorm_train_e694b0a25ea01d1f6cdc__45056_65536",
            "cake_rmsnorm_train_dca25ace64c075a81625__65536_inf",
            "cake_rmsnorm_train_628248febbdfe17c3325__320_1024",
            "cake_rmsnorm_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_45056",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__45056_65536",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_4",
            "cake_rmsnorm_train_744aa77460c886f4f429__4_16",
            "cake_rmsnorm_train_db8d986d719b909621d0__16_320",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__320_1024_107",
            "cake_rmsnorm_train_7c0dd63f0b011049603e__1024_16384",
            "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_45056",
            "cake_rmsnorm_train_0f0b782f1e538caad3ce__45056_65536",
            "cake_rmsnorm_train_ccd94d87e9036a0b2faf__65536_inf",
            "cake_rmsnorm_train_628248febbdfe17c3325__320_1024",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_45056",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__45056_65536",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_c5f62407b828a85f82ec",
            "cake_rmsnorm_train_a951623c157d070e85ab__4_16",
            "cake_rmsnorm_train_ef052c0918725ffa68d4__16_31",
            "cake_rmsnorm_train_32eb12e27be1360b890d__31_256",
            "cake_rmsnorm_train_845cdcd8145ada631461__256_1024",
            "cake_rmsnorm_train_81c3d8ab0a482e3af043",
            "cake_rmsnorm_train_845cdcd8145ada631461__16384_50176",
            "cake_rmsnorm_train_1a3df5ac020055477581__50176_50688",
            "cake_rmsnorm_train_694decfb79ba96468f62__50688_65536",
            "cake_rmsnorm_train_1a3df5ac020055477581__65536_inf",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50176",
            "cake_rmsnorm_train_c4d8be677fa70b841877__50688_65536",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_0017ab41f68d8689ebc2",
            "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_16",
            "cake_rmsnorm_train_22944704b77a80a04834__16_256",
            "cake_rmsnorm_train_838a9a4054731f7942af__256_1024",
            "cake_rmsnorm_train_810989bdea464df479ce__1024_16384",
            "cake_rmsnorm_train_838a9a4054731f7942af__16384_50176",
            "cake_rmsnorm_train_810989bdea464df479ce",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50176",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_945a0afdb82ab47eb643",
            "cake_rmsnorm_train_41f44b4bc0b765350a50__4_16",
            "cake_rmsnorm_train_44511fa02465a6d7283d__16_256",
            "cake_rmsnorm_train_a647fccc3d9fa9134733",
            "cake_rmsnorm_train_d41a514f5210e9adc39f",
            "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_24064",
            "cake_rmsnorm_train_99a01f2a850874718c3f__24064_24576",
            "cake_rmsnorm_train_aed93ac9e4828f1b8b47__24576_65536",
            "cake_rmsnorm_train_5e9d7baa36814bb7aaf6__65536_inf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_24064",
            "cake_rmsnorm_train_10e24516314c244771f4__24064_24576",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_f6bd1634a115bd281055",
            "cake_rmsnorm_train_b378ba007691c582171e__4_16",
            "cake_rmsnorm_train_440bd6f2a114414ea9db__16_256",
            "cake_rmsnorm_train_dc439f8335327e482025",
            "cake_rmsnorm_train_b08633208e60b8ce94fd",
            "cake_rmsnorm_train_0f1859b9fe9e47668a1c__4096_24064",
            "cake_rmsnorm_train_fa3d4fc5dbd76348b31a__24064_65536",
            "cake_rmsnorm_train_23cee954b648495ceef1__65536_inf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_24064",
            "cake_rmsnorm_train_10e24516314c244771f4__24064_65536",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_dbc680f356be4cd643de",
            "cake_rmsnorm_train_904b1ad45d6219b4f43c__4_12",
            "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__12_31",
            "cake_rmsnorm_train_505283a5225ac45bc095__31_256",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__256_511",
            "cake_rmsnorm_train_505283a5225ac45bc095__511_640",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__640_895",
            "cake_rmsnorm_train_505283a5225ac45bc095__895_896",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__896_4096",
            "cake_rmsnorm_train_6816fb80e97380f1bcbc",
            "cake_rmsnorm_train_b8b0913b40f035f43b76__19968_49152",
            "cake_rmsnorm_train_56634f53e6456b34f76f",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__256_511",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__640_895",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__896_4096",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_d47138228b123e55fe5d",
            "cake_rmsnorm_train_05396cd3a077d3772fdc__4_15",
            "cake_rmsnorm_train_3a7bd162c370762719b3__15_256",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__256_511",
            "cake_rmsnorm_train_3a7bd162c370762719b3__511_640",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__640_895",
            "cake_rmsnorm_train_3a7bd162c370762719b3__895_896",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__896_4096",
            "cake_rmsnorm_train_468cc4d1b8128187b4b3__4096_19968",
            "cake_rmsnorm_train_de0cf457cd240f9c6422__19968_32767",
            "cake_rmsnorm_train_9cdd6eef6888b2a1dd6a__32767_131072",
            "cake_rmsnorm_train_6cedccdf3a58f23f260f__131072_inf",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__256_511",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__640_895",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__896_4096",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_cdcd68d9227a76af4e88__19968_32767",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_e37752b056dd4e4a0bee",
            "cake_rmsnorm_train_fdfb73350c1a926a4929__4_16",
            "cake_rmsnorm_train_57d45549f8ed34e34e65__16_960",
            "cake_rmsnorm_train_5d292172df772adb17ae__960_1151",
            "cake_rmsnorm_train_57d45549f8ed34e34e65__1151_1152",
            "cake_rmsnorm_train_5d292172df772adb17ae__1152_19456",
            "cake_rmsnorm_train_19e18371f9243bd1a91a__19456_30720",
            "cake_rmsnorm_train_3edefef1a5b4b9561a13__30720_131072",
            "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82__131072_inf",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__960_1151",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__1152_19456",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_30720",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_deeb187630473818923d__0_4",
            "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_15",
            "cake_rmsnorm_train_c15868b87bfb2c21645e__15_320",
            "cake_rmsnorm_train_0dcdc563be377247044e__320_447",
            "cake_rmsnorm_train_c15868b87bfb2c21645e__447_576",
            "cake_rmsnorm_train_864e84f3d874f850c091__576_3584",
            "cake_rmsnorm_train_0dcdc563be377247044e__3584_7168",
            "cake_rmsnorm_train_16c5f0465183f87c5229__7168_131072",
            "cake_rmsnorm_train_c07ab5f16934c0cc8421__131072_inf",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__320_447",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
            "cake_rmsnorm_train_18b8b8054c7053942bcd__4_16",
            "cake_rmsnorm_train_a501b31feb94fa326722__16_1664",
            "cake_rmsnorm_train_44c5b870a7dcbb8de924__1664_2304",
            "cake_rmsnorm_train_e77387f14390ba486055__2304_19456",
            "cake_rmsnorm_train_b83f8b3d546630a60675__19456_28672",
            "cake_rmsnorm_train_25f9b466fa418dd2afc4__28672_inf",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_28672",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
            "cake_rmsnorm_train_c29a8f856283578103dc__4_16",
            "cake_rmsnorm_train_a1c12b0ee6454391a37f__16_768",
            "cake_rmsnorm_train_71d36617003ab7742a77__768_3584",
            "cake_rmsnorm_train_56f988dde54db37f09d9__3584_7168",
            "cake_rmsnorm_train_999d4acb1d1a4e0cbef0__7168_131072",
            "cake_rmsnorm_train_e655055ead76f8df4d24__131072_inf",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168",
        ],
    },
}


# The kernel-parameter plan of a record is one of three shapes -- the forward, the main backward and the
# dw reduce kernel -- shared by every record of that shape (``_CLASSES``); the shape follows from the
# record's role and workspace rule, so the records above carry only what differs between them.
def _plan_class(record: dict[str, Any]) -> str:
    if record["role"] == "bwd_reduce":
        return "bwd_reduce"
    if record["workspace_rule"] is not None:
        return "bwd_main"
    return "fwd"


for _record in MODULES.values():
    _record.update(_CLASSES[_plan_class(_record)])


HIDDEN_SIZES: tuple[int, ...] = (256, 512, 1536, 2048, 4096, 5120, 6144)

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
KINDS = ("fwd", "bwd")
# Row-strided BF16 inputs are read with 16-byte vector accesses; the forward
# launcher additionally requires 32-byte aligned row storage (its 256-bit load
# configuration), the backward 16 bytes.  Row strides are multiples of eight
# elements.
ROW_ALIGNMENT_ELEMENTS = 8
FORWARD_ALIGNMENT_BYTES = 32
BACKWARD_ALIGNMENT_BYTES = 16
# The weight dtypes a route may ship for: BF16 (every route) and FP32 (the
# ``_w32`` routes: the effective weight ``1 + w`` of a zero-centred model,
# computed by the caller in FP32).  The FP32-weight programs load ``w`` with
# 32-byte vectors in both directions, so the host reads an FP32 ``w`` in place
# when contiguous and 32-byte aligned and copies it once otherwise (a BF16
# ``w`` keeps the forward's 32-byte and the backward's 16-byte rule).
WEIGHT_DTYPES = ("bf16", "f32")
FP32_WEIGHT_ALIGNMENT_BYTES = 32
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "y",
    "r",
    "u",
    "h_new",
    "g",
    "g_h",
    "dx",
    "dw",
    "partial",
    "counters",
)
LOGICAL_SCALARS = (
    "rows",
    "eps",
    "x_stride",
    "u_stride",
    "g_stride",
    "g_h_stride",
    "n_chunks",
    "slots",
)


def route_key(hidden: int, kind: str, residual: bool, w_dtype: str = "bf16") -> str:
    """Name the route of one public call: ``<kind>[_w32][_residual]_h<hidden>``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if w_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"w_dtype must be one of {WEIGHT_DTYPES}, got {w_dtype!r}")
    w32 = "_w32" if w_dtype == "f32" else ""
    return f"{kind}{w32}{'_residual' if residual else ''}_h{int(hidden)}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes with a complete fwd + bwd route on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        hidden
        for hidden in HIDDEN_SIZES
        if all(route_key(hidden, kind, False) in routes for kind in KINDS)
    )


def weight_dtypes(arch: str, hidden: int, kind: str, residual: bool) -> tuple[str, ...]:
    """The weight dtypes (tokens of ``WEIGHT_DTYPES``) the route ships on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        w_dtype
        for w_dtype in WEIGHT_DTYPES
        if route_key(hidden, kind, residual, w_dtype) in routes
    )


def route_applies(
    *,
    device_capability: Sequence[int],
    hidden: int,
    kind: str,
    residual: bool,
    w_dtype: str = "bf16",
) -> bool:
    """Whether the verified export owns one training call."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return route_key(hidden, kind, residual, w_dtype) in ROUTES.get(arch, {})


def route_modules(
    arch: str, hidden: int, kind: str, residual: bool, w_dtype: str = "bf16"
) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(hidden, kind, residual, w_dtype)
    names = ROUTES.get(arch, {}).get(key)
    if not names:
        raise ValueError(
            f"unsupported Cake RMSNorm training route {key!r} on {arch!r}; exported "
            f"hidden sizes on this architecture: {supported_hidden_sizes(arch)}"
        )
    return tuple(names)


def _selects(rule: Optional[dict[str, Any]], rows: int) -> bool:
    if rule is None:
        return True
    kind = rule["kind"]
    if kind == "rows_between":
        at_most = rule["at_most"]
        return rows > int(rule["above"]) and (at_most is None or rows <= int(at_most))
    raise ValueError(f"unknown Cake RMSNorm training selection rule: {kind!r}")


def selected_modules(
    arch: str,
    hidden: int,
    kind: str,
    residual: bool,
    *,
    rows: int,
    w_dtype: str = "bf16",
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``rows`` tokens.

    A route may carry several programs of one kernel: the forward keeps
    latency programs for small and mid row counts next to its throughput
    program and switches on the row count.  Each module's recorded
    ``select_rule`` names the row range it owns (``rows_between``: ``above <
    rows <= at_most``, ``at_most`` ``None`` = unbounded; ``None`` = every call)
    and exactly the modules whose rule admits ``rows`` are launched, so the
    exported call runs the same program the source launcher selects.
    """

    rows = int(rows)
    names = tuple(
        name
        for name in route_modules(arch, hidden, kind, residual, w_dtype)
        if _selects(MODULES[name]["select_rule"], rows)
    )
    if not names:
        raise ValueError(
            f"no exported module of the {route_key(hidden, kind, residual, w_dtype)!r} "
            f"route on {arch!r} admits {rows} rows"
        )
    return names


def program_windows(
    arch: str, hidden: int, kind: str, residual: bool, w_dtype: str = "bf16"
) -> tuple[tuple[int, Optional[int], str], ...]:
    """The row-count windows of one route and the program each launches.

    One ``(above, at_most, library)`` per main program of the route (the dw
    reduce companions share their owner's window), from the smallest row
    counts up: the call with ``above < rows <= at_most`` (``at_most`` ``None``
    = unbounded) launches ``library``.  The windows tile every positive row
    count exactly once; a route whose records do not is refused.  Within one
    window a row's outputs do not depend on the token count (see the package
    docstrings); the edges are where they may change.
    """

    windows: list[tuple[int, Optional[int], str]] = []
    for name in route_modules(arch, hidden, kind, residual, w_dtype):
        record = MODULES[name]
        if record["role"] == "bwd_reduce":
            continue
        rule = record["select_rule"]
        if rule is None:
            windows.append((0, None, record["library"]))
            continue
        if rule["kind"] != "rows_between":
            raise ValueError(
                f"unknown Cake RMSNorm training selection rule: {rule['kind']!r}"
            )
        at_most = rule["at_most"]
        windows.append(
            (
                int(rule["above"]),
                None if at_most is None else int(at_most),
                record["library"],
            )
        )
    windows.sort(key=lambda window: window[0])
    key = route_key(hidden, kind, residual, w_dtype)
    above = 0
    for index, (low, high, _library) in enumerate(windows):
        last = index == len(windows) - 1
        if low != above or (high is None) != last or (high is not None and high <= low):
            raise ValueError(
                f"the windows of route {key!r} on {arch!r} do not tile the row counts: {windows}"
            )
        above = high if high is not None else above
    return tuple(windows)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // int(multiple)) * int(multiple)


# The backward workspace starts with the deterministic pairs' ``uint32``
# counter table (192 words at most, padded to four 256-byte lines); every pair
# kernel leaves the words it uses at zero.  The single kernel's counters
# (``SINGLE_COUNTERS_BYTES`` at ``SINGLE_COUNTERS_OFFSET``) and FP64 column sums
# (``SUMS_BYTES`` at ``SUMS_OFFSET``: the widest sums of the package's single
# records, their workspace rules' ``sum_bytes``) follow; the single kernel leaves
# its sums at zero and its counters in their rest state after every launch.
# The FP32 partial rows of the pairs' main kernels come last
# (``PAIR_ROWS_OFFSET``); a pair-row single kernel (``claim_part_f32``) binds the
# single counters and, at ``PAIR_ROWS_OFFSET``, scratch of one row of FP32 pairs
# per CTA (its rule's ``part_row_bytes``), which needs no rest state, like the
# pairs' rows.  Every offset is the package's, not the call's, so one
# buffer, zero when first used, serves calls of any token count, window, hidden
# size, residual flag and weight dtype with no host write between them, and the
# two forms never share a counter word.
COUNTERS_BYTES = 1024
SINGLE_COUNTERS_OFFSET = COUNTERS_BYTES
SINGLE_COUNTERS_BYTES = 256
SUMS_OFFSET = SINGLE_COUNTERS_OFFSET + SINGLE_COUNTERS_BYTES
# The workspace rule kinds of the backward forms: the main kernel of a
# deterministic pair, the single kernel, the pair-row single kernel.
WORKSPACE_KINDS = ("chunk_partials", "claim_sum_f64", "claim_part_f32")


def _sum_bytes(record: dict[str, Any]) -> int:
    """Bytes of the FP64 column sums a single-kernel record binds to
    ``partial`` (its workspace rule's ``sum_bytes``, the kernel's
    ``workspace_bytes`` of the build); 0 for another record."""

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "claim_sum_f64":
        return 0
    return int(rule["sum_bytes"])


SUMS_BYTES = _round_up(
    max((_sum_bytes(record) for record in MODULES.values()), default=0), 256
)
PAIR_ROWS_OFFSET = SUMS_OFFSET + SUMS_BYTES


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


# The chunk table of the main backward kernel.  TAIL_DIV and MAX_HALVING are
# the kernel helpers' own constants (a change there is a kernel text change and
# a change here): the halving stops once the next chunk would be at most
# ceil(share / tail_div) pipeline stages, and after at most max_halving halving
# rounds.  They are the default taper; a kernel built with a taper of its own
# records it under its grid rule ("tail_div" / "max_halving", each within the
# constant it is named after), and the host cuts that record's table with it
# (chunk_taper): the chunk count it passes must be the count the kernel wrote
# partial rows for.
TAIL_DIV = 12
MAX_HALVING = 5
# Rounds a table can have (the workspace bound): halving round k (k = 0, 1, ...) has chunks of
# ceil(share / 2 ** (k + 1)) rows and runs only while that exceeds
# ceil(share / TAIL_DIV), which no k with 2 ** (k + 1) >= TAIL_DIV does; the
# final equalised round adds at most ``slots`` chunks in its two parts.  At TAIL_DIV = 12
# that is k = 0, 1, 2 (2 ** 3 = 8 is below the divisor, 2 ** 4 = 16 is not): at most three
# halving rounds are ever cut, and with the final round a table holds at most 4 * slots chunks
# (MAX_HALVING = 5 does not bind), each of at least one stage of >= 1 row: n_chunks <= min(T, 4 * slots).
# A record's taper within the constants cuts no more halving rounds (a smaller
# divisor stops the halving sooner, a smaller cap caps it lower), so the bound
# holds the table of every record.
MAX_ROUNDS = min(MAX_HALVING, max(0, (TAIL_DIV - 1).bit_length() - 1)) + 1


def chunk_rounds(
    T: int,
    slots: int,
    rpi: int,
    tail_div: int = TAIL_DIV,
    max_halving: int = MAX_HALVING,
) -> list[tuple[int, int]]:
    """``[(count, rows)]`` of the backward chunk table of ``T`` tokens, in claim order.

    Cut in pipeline stages of ``rpi`` rows: with ``S = ceil(T / rpi)`` stages,
    ``sl = clamp(slots, 1, S)`` and ``share = ceil(S / sl)`` stages per slot,
    halving rounds of ``sl`` chunks of ``ceil(share / 2)``, ``ceil(share / 4)``,
    ... stages while the size is above ``max(1, ceil(share / tail_div))``, the
    round fits the stages left and fewer than ``max_halving`` ran, then one
    final round spreading the stages left evenly over ``sl`` chunks (the first
    ``m`` one stage longer).  ``rows`` is ``stages x rpi``; :func:`chunk_table`
    ends the last chunk at ``T``.  Rounds of zero stages are not emitted; with
    ``rpi = 1`` the stages are rows.  ``tail_div`` / ``max_halving`` are the
    kernel's taper: the module constants unless its record spells its own
    (:func:`chunk_taper`).  The main backward kernel's CTAs claim the chunks
    from a counter in this order and the kernels compute the same table in
    integer arithmetic from ``T``, ``slots``, their rows per step and their
    taper.  Under a deterministic pair the FP32 partial row of every chunk, and
    the ``dw`` the reduce kernel folds from them in chunk order, are pure
    functions of ``(T, slots, rpi, tail_div, max_halving)``; the single kernel
    adds the chunks' sums into FP64 column sums in the order its CTAs finish,
    so its ``dw`` depends on scheduling (its ``dx`` does not).
    """

    if tail_div < 1 or max_halving < 0:
        raise ValueError(
            f"tail_div must be >= 1 and max_halving >= 0, got {tail_div} / {max_halving}"
        )
    S = -(-T // rpi)
    sl = max(1, min(slots, S))
    share = -(-S // sl)
    tail = max(1, -(-share // tail_div))
    rounds: list[tuple[int, int]] = []
    rem, s = S, -(-share // 2)
    while len(rounds) < max_halving and s > tail:
        if sl * s > rem:
            break
        rounds.append((sl, s * rpi))
        rem -= sl * s
        s = -(-s // 2)
    q, m = divmod(rem, sl)
    if m:
        rounds.append((m, (q + 1) * rpi))
    if q:
        rounds.append((sl - m, q * rpi))
    return rounds


def chunk_table(
    T: int,
    slots: int,
    rpi: int,
    tail_div: int = TAIL_DIV,
    max_halving: int = MAX_HALVING,
) -> list[tuple[int, int]]:
    """``[(begin, end)]`` of every chunk in claim order: covers ``[0, T)`` exactly, no overlap."""

    table, b = [], 0
    for count, rows in chunk_rounds(T, slots, rpi, tail_div, max_halving):
        for _ in range(count):
            e = min(b + rows, T)
            table.append((b, e))
            b = e
    return table


@functools.lru_cache(maxsize=4096)
def n_chunks(
    T: int,
    slots: int,
    rpi: int,
    tail_div: int = TAIL_DIV,
    max_halving: int = MAX_HALVING,
) -> int:
    """Chunks of the backward chunk table of ``T`` tokens (see :func:`chunk_rounds`).

    Cached per ``(T, slots, rpi, tail_div, max_halving)``: a training step
    repeats its token counts, and the per-call host path is this lookup, the
    main launch (``slots`` its twelfth argument) and, under a deterministic
    pair, the reduce launch (``n_chunks``).
    """

    return len(chunk_table(T, slots, rpi, tail_div, max_halving))


def chunk_taper(rule: dict[str, Any]) -> tuple[int, int]:
    """``(tail_div, max_halving)`` of a chunk-claiming grid rule: the taper its
    kernel was built with, which the host cuts the record's table with.

    Every ``claimed_chunks`` record spells both (the module constants for a
    kernel at the default taper), each an integer within the constant it is
    named after -- ``1 <= tail_div <= TAIL_DIV`` and ``0 <= max_halving <=
    MAX_HALVING`` -- so :func:`chunk_capacity` bounds its table.  A rule that
    lacks one or exceeds a bound is refused rather than cut at the defaults:
    that would pass the reduce another chunk count than the main kernel wrote
    partial rows for.
    """

    taper = []
    for name, bound, floor in (
        ("tail_div", TAIL_DIV, 1),
        ("max_halving", MAX_HALVING, 0),
    ):
        value = rule.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"grid rule {rule.get('kind')!r} spells no integer {name} (found "
                f"{value!r}); every chunk-claiming record carries its kernel's taper"
            )
        if not floor <= value <= bound:
            raise ValueError(
                f"grid rule {name} {value} is outside [{floor}, {bound}] (the module "
                f"constant {name.upper()} bounds it)"
            )
        taper.append(value)
    return taper[0], taper[1]


def chunk_capacity(rows: int, slots: int) -> int:
    """Partial rows a workspace holds for every token count up to ``rows`` on ``slots`` slots.

    ``min(rows, MAX_ROUNDS * slots)``: a table has at most :data:`MAX_ROUNDS`
    rounds of at most ``slots`` chunks whatever the rows per step and the
    taper (a record's taper stays within the module constants, which cut the
    most rounds), and every chunk holds a row; the bound grows with ``rows``,
    so a buffer sized for ``rows`` serves every smaller token count too.
    """

    rows, slots = int(rows), int(slots)
    if rows <= 0 or slots <= 0:
        raise ValueError("rows and slots must be positive")
    return min(rows, MAX_ROUNDS * slots)


def table_slots(rule: dict[str, Any], *, device_index: int) -> int:
    """Chunk-table slots of a ``claimed_chunks`` grid rule on the device.

    ``table_slots_per_sm * SM count`` when the rule spells
    ``table_slots_per_sm`` (a build whose table is cut for more, shorter
    chunks than its launch has CTAs: each CTA claims several), else the
    launch's own ``ctas_per_sm * SM count``.  The main kernel takes it as its
    ``slots`` argument; the grid stays bounded by ``ctas_per_sm * SM count``.
    """

    per_sm = int(rule.get("table_slots_per_sm", 0)) or int(rule["ctas_per_sm"])
    return per_sm * sm_count(device_index)


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``rows_per_cta`` modules launch ``ceil(rows / rows_per_cta)`` CTAs, capped
    at ``grid_limit`` when the rule records a positive limit, else at
    ``grid_per_sm * SM count`` of the device when that is positive (persistent
    grid-stride row loop in the kernel).  ``fixed`` modules launch exactly
    ``ctas`` CTAs (the dw reduce kernel: one CTA per column slab; the
    one-cluster backward kernel: the CTAs of its one thread-block cluster, the
    cluster dimension its binding launches with).
    ``claimed_chunks`` modules are the main backward kernel: up to
    ``ctas_per_sm * SM count`` persistent CTAs claim the chunks of the table of
    :func:`chunk_rounds` built on :func:`table_slots` slots (``rows_per_issue``
    rows per pipeline step of the kernel, under the taper its record spells:
    :func:`chunk_taper`), so the grid is ``min(ctas_per_sm * SM count,
    n_chunks)``; a column-task kernel's rule spells its ``col_tasks`` (``H /
    col`` column tasks, tickets beside the chunks), so its grid is
    ``min(ctas_per_sm * SM count, n_chunks + col_tasks)`` and a small ``T``
    runs every task at once.  The launch is a plain
    one -- no co-residency is required -- and ``ctas_per_sm`` is the production
    grid factor capped at the launch bound the kernel declares.  ``rows`` must
    be positive; the public API returns empty outputs for zero rows without
    launching.
    """

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "rows_per_cta":
        grid_x = _ceil_div(rows, int(rule["rows_per_cta"]))
        limit = int(rule.get("grid_limit", 0))
        per_sm = int(rule.get("grid_per_sm", 0))
        if limit <= 0 and per_sm > 0:
            limit = per_sm * sm_count(device_index)
        if limit > 0:
            grid_x = min(grid_x, limit)
    elif kind == "fixed":
        grid_x = int(rule["ctas"])
    elif kind == "claimed_chunks":
        slots = table_slots(rule, device_index=device_index)
        grid_x = min(
            int(rule["ctas_per_sm"]) * sm_count(device_index),
            n_chunks(rows, slots, int(rule["rows_per_issue"]), *chunk_taper(rule))
            + int(rule.get("col_tasks", 0)),
        )
    else:
        raise ValueError(f"unknown Cake RMSNorm training grid rule: {kind!r}")
    return (max(grid_x, 1), 1, 1)


def chunking(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int]:
    """``(n_chunks, slots)`` of the backward partition for ``rows`` tokens.

    The main backward module's CTAs claim the ``n_chunks`` chunks of the table
    built on ``slots`` = :func:`table_slots` under the module's taper
    (:func:`chunk_taper`); both are host-known and never depend on scheduling.
    The main kernel takes both; the dw reduce kernel a deterministic pair
    launches after it takes ``n_chunks``.  A one-cluster module (grid rule
    ``fixed``: its CTAs are one thread-block cluster over every row) reads
    neither: ``(1, 1)``.
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] not in WORKSPACE_KINDS:
        raise ValueError(f"module kernel_{record['kernel']} owns no backward workspace")
    grid = record["grid_rule"]
    if grid["kind"] == "fixed":
        # The one-cluster kernel: its ``ctas`` CTAs (one thread-block cluster) cover
        # every row and read neither count; one chunk of one slot.
        return 1, 1
    if grid["kind"] != "claimed_chunks":
        raise ValueError(
            f"module kernel_{record['kernel']} launches under {grid['kind']!r}, not a "
            "chunk-claiming grid"
        )
    slots = table_slots(grid, device_index=device_index)
    return n_chunks(rows, slots, int(grid["rows_per_issue"]), *chunk_taper(grid)), slots


def partial_capacity(record: dict[str, Any], rows: int, *, device_index: int) -> int:
    """Partial rows the workspace of one main backward module holds for every token count up to ``rows``."""

    rule = record["grid_rule"]
    if rule["kind"] != "claimed_chunks":
        raise ValueError(f"grid rule {rule['kind']!r} has no chunk table")
    return chunk_capacity(rows, table_slots(rule, device_index=device_index))


def workspace_layout(
    record: dict[str, Any],
    rows: int,
    *,
    device_index: int,
    capacity_record: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module.

    The workspace is one CUDA ``uint8`` buffer: the deterministic pairs'
    ``uint32`` counters in its first :data:`COUNTERS_BYTES` bytes, the single
    kernel's counters at :data:`SINGLE_COUNTERS_OFFSET`, its FP64 column sums at
    :data:`SUMS_OFFSET` and the pairs' FP32 partial rows at
    :data:`PAIR_ROWS_OFFSET`.  A single-kernel module (workspace rule
    ``claim_sum_f64``) binds the single counters and its rule's ``sum_bytes``
    of sums, whatever ``rows``.  A pair-row single-kernel module (workspace
    rule ``claim_part_f32``) binds the single counters and, at
    :data:`PAIR_ROWS_OFFSET`, ``part_row_bytes`` of FP32 scratch per CTA of its
    launch grid (:func:`launch_grid`, returned as ``grid``; the chunk table's
    ``slots`` may differ from it); a buffer sized for ``rows`` holds a row for
    every CTA of any launch up to ``rows`` (``min(rows, ctas_per_sm * SM
    count)`` rows, the bound of the launch grid).  The main kernel of a
    deterministic pair
    (``chunk_partials``) binds the pair counters and the FP32
    ``[capacity, hidden]`` per-chunk partial rows, ``capacity`` the
    :func:`chunk_capacity` of ``rows``, so a buffer sized for ``rows`` serves
    every smaller token count.  The buffer is zero when first used and every
    kernel leaves its counters in their rest state and the sums at zero, so no
    host write is needed between launches, layouts or CUDA Graph replays.  A
    route whose pair modules switch on the row count shares one buffer:
    ``capacity_record`` (the pair main module with the largest capacity) fixes
    the partial capacity for every pair module of the route, while
    ``n_chunks`` and ``slots`` follow ``record``, the module launched.
    """

    rule = record["workspace_rule"]
    chunks, slots = chunking(record, rows, device_index=device_index)
    hidden = int(record["hidden"])
    counters = int(rule["counters"])
    if rule["kind"] == "claim_sum_f64":
        if 4 * max(counters, 1) > SINGLE_COUNTERS_BYTES:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {counters} counters; the "
                f"single kernel's table holds {SINGLE_COUNTERS_BYTES // 4}"
            )
        partial_bytes = _sum_bytes(record)
        if partial_bytes % 8 or partial_bytes > SUMS_BYTES:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {partial_bytes} bytes of "
                f"FP64 sums; the workspace holds {SUMS_BYTES}"
            )
        return {
            "n_chunks": chunks,
            "slots": slots,
            "capacity_chunks": 0,
            "partial_bytes": partial_bytes,
            "partial_offset": SUMS_OFFSET,
            "counters": counters,
            "counters_offset": SINGLE_COUNTERS_OFFSET,
            "total_bytes": SUMS_OFFSET + _round_up(partial_bytes, 256),
        }
    if rule["kind"] == "claim_part_f32":
        if 4 * max(counters, 1) > SINGLE_COUNTERS_BYTES:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {counters} counters; the "
                f"single kernel's table holds {SINGLE_COUNTERS_BYTES // 4}"
            )
        row_bytes = int(rule["part_row_bytes"])
        if row_bytes <= 0 or row_bytes % 8:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {row_bytes} part_row_bytes; "
                "one row of FP32 pairs is a positive multiple of 8 bytes"
            )
        grid = launch_grid(record, rows, device_index=device_index)[0]
        ctas = int(record["grid_rule"]["ctas_per_sm"]) * sm_count(device_index)
        return {
            "n_chunks": chunks,
            "slots": slots,
            "grid": grid,
            "capacity_chunks": 0,
            "partial_bytes": row_bytes * grid,
            "partial_offset": PAIR_ROWS_OFFSET,
            "counters": counters,
            "counters_offset": SINGLE_COUNTERS_OFFSET,
            "total_bytes": PAIR_ROWS_OFFSET
            + _round_up(row_bytes * min(int(rows), ctas), 256),
        }
    if 4 * max(counters, 1) > COUNTERS_BYTES:
        raise ValueError(
            f"module kernel_{record['kernel']} declares {counters} counters; the "
            f"workspace holds {COUNTERS_BYTES // 4}"
        )
    capacity = partial_capacity(
        record if capacity_record is None else capacity_record,
        rows,
        device_index=device_index,
    )
    if capacity < chunks:
        raise ValueError(
            f"the route's partial capacity {capacity} is below the {chunks} chunks of "
            f"module kernel_{record['kernel']} for {rows} rows"
        )
    partial_bytes = 4 * capacity * hidden
    return {
        "n_chunks": chunks,
        "slots": slots,
        "capacity_chunks": capacity,
        "partial_bytes": partial_bytes,
        "partial_offset": PAIR_ROWS_OFFSET,
        "counters": counters,
        "counters_offset": 0,
        "total_bytes": PAIR_ROWS_OFFSET + _round_up(partial_bytes, 256),
    }


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake RMSNorm training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake RMSNorm training module {name} is not delivered for {arch}; "
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


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid follows
    the module's recorded grid rule for ``rows`` tokens and the kernel runs on
    the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, rows, device_index=device_index)
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
        module.run(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BACKWARD_ALIGNMENT_BYTES",
    "COUNTERS_BYTES",
    "FORWARD_ALIGNMENT_BYTES",
    "FP32_WEIGHT_ALIGNMENT_BYTES",
    "HIDDEN_SIZES",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_HALVING",
    "MAX_ROUNDS",
    "MODULES",
    "ROUTES",
    "PAIR_ROWS_OFFSET",
    "ROW_ALIGNMENT_ELEMENTS",
    "SINGLE_COUNTERS_BYTES",
    "SINGLE_COUNTERS_OFFSET",
    "SUMS_BYTES",
    "SUMS_OFFSET",
    "TAIL_DIV",
    "WEIGHT_DTYPES",
    "WORKSPACE_KINDS",
    "arch_for_capability",
    "chunk_capacity",
    "chunk_rounds",
    "chunk_table",
    "chunking",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "n_chunks",
    "partial_capacity",
    "program_windows",
    "route_applies",
    "route_key",
    "route_modules",
    "run_module",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "supported_hidden_sizes",
    "weight_dtypes",
    "workspace_layout",
]
