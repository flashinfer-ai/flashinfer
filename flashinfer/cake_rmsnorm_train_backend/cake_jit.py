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
sums, so ``dw`` is not guaranteed bitwise reproducible.  The deterministic
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
    # sm_100a H256 bf16 weight: (0, 20480] backward time 0.804-1.231x the det pair's at 4 to 16384 rows over two to ten timing sessions per row count, above 1 only in one session at 5112 and 5124 rows (B200)
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
            "above": 0,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H256 f32 weight: (0, 20480] backward time 0.762-1.052x the det pair's at 4 to 16384 rows over two to six timing sessions per row count, above 1 only in one session at 5112, 5120 and 5124 rows (B200)
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
            "above": 0,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H512 bf16 weight: (1280, inf] FI bench ratio < 1 against the det pair in both sessions (B200)
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 16384,
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
    # sm_100a H1536 bf16 weight: (36864, 78784] FI bench ratio < 1 against the det pair in both sessions (B200); up to T 78784 its worst dw element stays within 0.877 of the per-element budget at every 64th T from 36864 (worst of 240 inputs of each of three families in both rstd forms) on B200; it exceeds the budget at larger T (1.029 at T 118528 and 2.402 at T 1048576, worst of 120 inputs of each of three families); the twosum form of this kernel runs above T 78784
    "cake_rmsnorm_train_ae610afdf77ff6714a5b__36864_78784": {
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
        "kernel": "cake_rmsnorm_train_ae610afdf77ff6714a5b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ae610afdf77ff6714a5b",
        "role": "bwd_main",
        "select_rule": {
            "above": 36864,
            "at_most": 78784,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
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
    # sm_100a H6144 bf16 weight: (5632, 65536] FI bench ratio < 1 against the det pair in both sessions (B200); up to T 65536 its worst dw element stays within 0.751 of the per-element budget at every sampled T from 5633 to 65536 (75 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 72 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_9433061da18b2db4299d__5632_65536": {
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
            "above": 5632,
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
    # sm_100a H6144 residual bf16 weight: (1408, 65536] FI bench ratio < 1 against the det pair in both sessions (B200); up to T 65536 its worst dw element stays within 0.718 of the per-element budget at every sampled T from 1409 to 65536 (75 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 72 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_3fbfdd750cbeb47032b5__1408_65536": {
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
        "kernel": "cake_rmsnorm_train_3fbfdd750cbeb47032b5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3fbfdd750cbeb47032b5",
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
    # sm_100a H2048 bf16 weight: (0, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.32x to 1.59x the det package's speed and 1.33x to 1.63x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_ef052c0918725ffa68d4__0_64": {
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
            "above": 0,
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
    # sm_100a H2048 f32 weight: (0, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.39x to 1.53x the det package's speed and 1.35x to 1.56x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64": {
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
            "above": 0,
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
    # sm_100a H4096 bf16 weight: (0, 64] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.27x to 1.49x the det package's speed and 1.09x to 1.30x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_231e01eca0459c2c0bf5__0_64": {
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
            "above": 0,
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
    # sm_100a H4096 f32 weight: (0, 64] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.20x to 1.40x the det package's speed and 1.13x to 1.31x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_ce1f7d216ecc0a1573b9__0_64": {
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
            "above": 0,
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
    # sm_100a H5120 bf16 weight: (0, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.18x to 1.41x the det package's speed and 1.16x to 1.42x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__0_64": {
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
            "above": 0,
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
    # sm_100a H6144 bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.42x to 2.09x the det package's speed and 1.21x to 1.51x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_d70f2562a592151ec4f9__0_64": {
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
            "above": 0,
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
    # sm_100a H6144 f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.37x to 2.15x the det package's speed and 1.22x to 1.43x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_9b37923697ee8ca52a9e__0_64": {
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
            "above": 0,
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
    # sm_100a H6144 residual bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.31x to 2.04x the det package's speed and 1.24x to 1.51x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_7b0f5a6a8769473c322e__0_64": {
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
            "above": 0,
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
    # sm_100a H6144 residual f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.32x to 2.10x the det package's speed and 1.25x to 1.46x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_ac9d69f94aabfc68aa40__0_64": {
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
            "above": 0,
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
    # sm_100a H4096 bf16 weight: (4608, 36864] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_15e157d2ee9080d206a9": {
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
        "kernel": "cake_rmsnorm_train_15e157d2ee9080d206a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_15e157d2ee9080d206a9",
        "role": "bwd_main",
        "select_rule": {
            "above": 4608,
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
    # sm_100a H2048 bf16 weight: (45056, 58880] the single kernel of its window above, within 2 % of the det pair's backward time (the det pair of the window, the deterministic package's program, reaches 0.969 of the per-element budget at T 58560, K form, worst of 120 inputs of each of three families at every 64th T of the window on B200); this single kernel's worst dw element stays within 0.800 of the per-element budget at every 64th T of the window (worst of 120 inputs of each of three families in both rstd forms) on B200; backward time 0.99x to 1.02x the det pair's at T 45057, 51968 and 58880 (B200)
    "cake_rmsnorm_train_0fa4f255caf709d09f66__45056_58880": {
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
    # sm_100a H2048 bf16 weight: (13608, 45056] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_0fa4f255caf709d09f66__13608_45056": {
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
            "above": 13608,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
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
    # sm_103a H5120 bf16 weight: (61952, inf] above T 61952 for dw accuracy (both weight dtypes switch at T 61952: the FP32-weight route's compensated det pair, the same dw computation, reaches 0.888 of the per-element budget at T 62464 on GB300); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.588 of the budget at the sampled T from 61953 to 131072 on GB300 (3 x 60 inputs, kernel and torch rstd); its FP32-weight twin, the same dw computation, 0.688 at 73 sampled T from 61953 to 131072, 1.16 times the twosum single's speed in the FI bench (GB300, 4 sessions)
    "cake_rmsnorm_train_b523a764b029e5ca1aaf": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 81920,
        },
    },
    # sm_103a H5120 f32 weight: (61952, inf] above T 61952 for dw accuracy (the compensated det pair, from T 5120, reaches 0.888 of the per-element budget at T 62464 on GB300); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.688 of the budget at 73 sampled T from 61953 to 131072 on GB300 (3 x 60 inputs per T, kernel and torch rstd), at most the FP32 eager reference's, 1.18 times the twosum single's speed in the FI bench (GB300, 4 sessions)
    "cake_rmsnorm_train_90598c14b4e396f5c8b0": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 81920,
        },
    },
    # sm_103a H6144 residual bf16 weight: (2816, 4608] FI bench ratio < 1 against the det pair in both sessions (GB300)
    "cake_rmsnorm_train_115dcc719a6de4b079f1": {
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
        "kernel": "cake_rmsnorm_train_115dcc719a6de4b079f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_115dcc719a6de4b079f1",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 2816,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 residual bf16 weight: (5632, 32768] FI bench ratio < 1 against the det pair in both sessions (GB300)
    "cake_rmsnorm_train_115dcc719a6de4b079f1__5632_32768": {
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
        "kernel": "cake_rmsnorm_train_115dcc719a6de4b079f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_115dcc719a6de4b079f1",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 5632,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 residual f32 weight: (3584, 4608] FI bench ratio < 1 against the det pair in both sessions (GB300)
    "cake_rmsnorm_train_3c390339fa8c82311a0a": {
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
        "kernel": "cake_rmsnorm_train_3c390339fa8c82311a0a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c390339fa8c82311a0a",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 3584,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 residual f32 weight: (5632, 15872] FI bench ratio < 1 against the det pair in both sessions (GB300)
    "cake_rmsnorm_train_3c390339fa8c82311a0a__5632_15872": {
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
        "kernel": "cake_rmsnorm_train_3c390339fa8c82311a0a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c390339fa8c82311a0a",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 5632,
            "at_most": 15872,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    "cake_rmsnorm_train_b3e6531e46667901ab91__320_448": {
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
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 320,
            "at_most": 448,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (0, 48] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.08x to 1.47x the det package's speed and 1.10x to 1.49x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_7b0f5a6a8769473c322e__0_48": {
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
            "above": 0,
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
    # sm_107a H256 bf16 weight: (3584, 36864] backward time 0.893-0.937x the det pair's at 16384 rows, 0.981-1.072x at 4096, 24576, 32768 and 32772 rows, 1.025-1.142x at 6144 and 6400 rows; 4 to 10 timing sessions per row count (R200, 212 SMs)
    "cake_rmsnorm_train_195c4317b51bb4ea1137__3584_36864": {
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
        "kernel": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "role": "bwd_main",
        "select_rule": {
            "above": 3584,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H6144 residual bf16 weight: (7168, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces exceeds the per-element budget at T 127424 (1.014, measured with FP32 weight; the BF16-weight program computes the same dw) on R200 with 212 SMs; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 7169 to 131072 (65 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.96x to 0.98x the single kernel's at T 7169, 32768, 65536, 98304 and 131072 in two sessions (R200, 212 SMs); faster than the deterministic package's program for those token counts too (0.95x to 0.99x its time at the same T)
    "cake_rmsnorm_train_c07ab5f16934c0cc8421__7168_inf": {
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
            "above": 7168,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_107a H6144 residual f32 weight: (7168, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the worst dw element of the single kernel it replaces exceeds the per-element budget at T 127424 (1.014) on R200 with 212 SMs; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 7169 to 131072 (131 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.96x to 0.99x the single kernel's at T 14337, 32768, 65536, 98304 and 131072 in two sessions (R200, 212 SMs); faster than the deterministic package's program for those token counts too (0.95x to 0.97x its time at the same T)
    "cake_rmsnorm_train_e655055ead76f8df4d24__7168_inf": {
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
            "above": 7168,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_107a H6144 residual bf16 weight: (448, 3584] backward time 1.009-1.046x the det pair's at T 1024 in four timing sessions (R200, 212 SMs)
    "cake_rmsnorm_train_864e84f3d874f850c091__448_3584": {
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
            "above": 448,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (448, 3584] backward time 1.039-1.050x the det pair's at T 1024 in four timing sessions (R200, 212 SMs)
    "cake_rmsnorm_train_71d36617003ab7742a77__448_3584": {
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
            "above": 448,
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
    # sm_100a H256 f32 weight: (524288, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T (from 92160 to 524288 rows the deterministic package's det pair runs, one cut with the BF16-weight route: its worst dw element stays within 0.69 of the per-element budget at the sampled row counts from 80896 to 524288 rows on B200, worst of 60 inputs of each of three families)
    "cake_rmsnorm_train_7c2aa4b429538da06a2a": {
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
    "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__92160_524288": {
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
            "at_most": 524288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H6144 residual bf16 weight: (32768, inf] the twosum form for the open top window: exact per-thread dw sums, so the dw error does not grow with T, within 3 % of the plain form's time
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
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 residual f32 weight: (15872, inf] above T 15872 for dw accuracy (the FP64-row single's worst dw error exceeds the FP32 eager reference's at T 16384 on GB300); twosum: exact per-thread dw sums
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
            "above": 15872,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H1536 bf16 weight: (46592, inf] above T 46592 for dw accuracy (the same dw computation as the FP32-weight route's det pair, whose worst dw element reaches 0.890 of the per-element budget at T 47104 on GB300; this route's det pair reaches 0.886 at T 53760 and exceeds the budget at T 55296, 1.016 of it); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_327bfeb4ab8fefc2731d": {
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
            "above": 46592,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 f32 weight: (37888, inf] above T 37888 for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 38400 on GB300, 0.7399 against 0.7092 of the per-element budget; its worst dw element reaches 0.890 of the budget at T 47104 and 0.993 at T 56832); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_963a7ca8d03f78bbe6f9": {
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
            "above": 37888,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H4096 bf16 weight: (50176, inf] above T 50176 for dw accuracy (the det pair's worst dw element exceeds the per-element budget at T 50688 on GB300, 1.032 of it, and reaches 0.989 of it at T 57856); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.608 of the budget at the sampled T from 61953 to 131072 on GB300 (3 x 60 inputs, kernel and torch rstd); its FP32-weight twin, the same dw computation, 0.670 at 71 sampled T from 9217 to 131072, 1.01 times the twosum single's speed in the FI bench (GB300, 4 sessions)
    "cake_rmsnorm_train_8250eb924d925c0db0d9": {
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
        "kernel": "cake_rmsnorm_train_8250eb924d925c0db0d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8250eb924d925c0db0d9",
        "role": "bwd_main",
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
    # sm_107a H256 bf16 weight: (36864, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above this edge
    "cake_rmsnorm_train_172a47a15d4c1a76c061__36864_inf": {
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
            "above": 36864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H256 f32 weight: (36864, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above this edge
    "cake_rmsnorm_train_7c2aa4b429538da06a2a__36864_inf": {
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
            "above": 36864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H4096 bf16 weight: (36864, 65536] above T 36864 for dw accuracy (the det pair's worst dw element reaches 0.900 of the per-element budget at T 38912 on R200 with 216 SMs); up to T 65536: its worst dw element stays within 0.734 of the per-element budget at every sampled T from 36865 to 65536 (112 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 111 of them) on R200 with 212 SMs, and exceeds it at T 119040 (1.181, worst of 120 inputs of each of three families in both rstd forms); the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_f11a86167db4f8e9a5da__36864_65536": {
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
        "kernel": "cake_rmsnorm_train_f11a86167db4f8e9a5da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f11a86167db4f8e9a5da",
        "role": "bwd_main",
        "select_rule": {
            "above": 36864,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
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
    # sm_107a H4096 f32 weight: (24064, 65536] above T 24064 for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 24576, 0.7328 against 0.7241 of the per-element budget, and at T 29696 on R200 with 212 SMs); up to T 65536: its worst dw element stays within 0.784 of the per-element budget at every sampled T from 24065 to 65536 (206 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 191 of them) on R200 with 212 SMs, and exceeds it at T 114422 (1.211, worst of 120 inputs of each of three families in both rstd forms); the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_a1f537fc94bde175bb3c__24064_65536": {
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
        "kernel": "cake_rmsnorm_train_a1f537fc94bde175bb3c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a1f537fc94bde175bb3c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24064,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
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
    # sm_107a H6144 bf16 weight: (30720, inf] the twosum form of the single kernel (its knobs plus twosum; twosum: exact per-thread dw sums) for dw accuracy up to T 131072, the top of the qualified range: the deterministic package's program for those token counts, faster here, exceeds the per-element budget at T 131072 (1.234) on R200 with 212 SMs, and its FP32-weight twin (the same dw computation) at T 98826 (1.098, worst of 120 inputs of each of three families in both rstd forms); the single kernel it replaces exceeds it at T 129472 (1.188, measured with FP32 weight; the BF16-weight program computes the same dw); the window keeps its start at T 30720 until a follow-up has qualifying evidence for the deterministic package's program above it; its worst dw error stays within 0.0006 of the per-element budget at every sampled T from 30721 to 131072 (65 T) on R200, worst of 60 inputs of each of three families in both rstd forms; backward time 0.78x to 0.81x the single kernel's at T 30721, 65536, 98304 and 131072 in two sessions (R200, 212 SMs)
    "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82__30720_inf": {
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
            "above": 30720,
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
    # sm_103a H4096 f32 weight: (9216, inf] above T 9216 for dw accuracy (over inputs 0-59 at the forward's rstd the det pair's worst error exceeds the FP32 eager reference's at T 9728 on GB300, 0.3383 against 0.3332, and at T 24576, 0.6127 against 0.6017); kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs): worst dw element 0.670 of the budget at 71 sampled T from 9217 to 131072 on GB300 (3 x 60 inputs per T, kernel and torch rstd), at most the FP32 eager reference's, 1.01 to 1.07 times the twosum single's speed in the FI bench (GB300, 4 sessions)
    "cake_rmsnorm_train_2911faf7a786144495a6": {
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
            "above": 9216,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_103a H6144 bf16 weight: (49152, inf] above T 49152 for dw accuracy (both weight dtypes switch at T 49152: the FP32-weight route's det pair, the same dw computation, reaches 0.868 of the per-element budget at T 49153 on GB300 for an input both routes draw; this route's det pair exceeds the budget at T 55296, 1.091 of it); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_582b2a20f9be648af8ec": {
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
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_103a H6144 f32 weight: (49152, inf] above T 49152 for dw accuracy (the det pair's worst dw element reaches 0.868 of the per-element budget at T 49153 on GB300 and exceeds it at T 54273, 1.085 of it in the kernel-tests harness, and at T 56320, 1.010 of it); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_4f9a76e68671f213ab7c": {
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
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 98304,
        },
    },
    # sm_107a H2048 bf16 weight: (50176, inf] above T 50176 for dw accuracy (the det pair's worst dw element reaches 0.900 of the per-element budget at T 51200 on R200 with 212 SMs); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_1a3df5ac020055477581": {
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
    # sm_107a H1536 bf16 weight: (53248, 65536] above T 53248 for dw accuracy (the FP32-weight route's det pair, the same dw computation, reaches 0.895 of the per-element budget at T 53760 and exceeds it at T 61952 (1.020) on R200 with 212 SMs; this route's det pair reaches 0.889 at T 54784 and 0.980 at T 62976); the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_3f75f4a81f2fb8269389__53248_65536": {
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
        "kernel": "cake_rmsnorm_train_3f75f4a81f2fb8269389",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3f75f4a81f2fb8269389",
        "role": "bwd_main",
        "select_rule": {
            "above": 53248,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
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
    # sm_107a H1536 f32 weight: (53248, 65536] above T 53248 for dw accuracy (the det pair's worst dw element reaches 0.895 of the per-element budget at T 53760 on R200 with 212 SMs and exceeds the budget at T 61952, 1.020 of it); the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_7c0dd63f0b011049603e__53248_65536": {
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
            "above": 53248,
            "at_most": 65536,
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
    # sm_100a H2048 bf16 weight: (58880, 65536] above T 58880 for dw accuracy (the det pair's worst dw element reaches 0.883 of the per-element budget at T 59392 on B200); up to T 65536 its worst dw element stays within 0.827 of the per-element budget at every sampled T from 58881 to 65536 (105 T, every multiple of 64 rows from 58944 to 65536 among them; worst of 60 inputs of each of three families in both rstd forms at 65 of them, and of 120 inputs at every multiple of 64 rows, worst 0.753 there) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_0fa4f255caf709d09f66__58880_65536": {
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
            "above": 58880,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 bf16 weight: (65536, 131072]; kblock single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs; producer warp, pairs in the shared row, evict_last loads, 3 stages): it replaces the twosum single 83f905ff for speed up to T 131072; worst dw element 0.7531 of the budget at the sampled T from 65537 to 131072 on B200 (3 x 120 inputs per T, kernel and torch rstd); above T 131072, beyond the sampled token counts, the twosum single 83f905ff keeps the window; backward time 0.86 to 0.87 times the replaced single's (B200)
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
    # sm_100a H2048 f32 weight: (58880, 65536] above T 58880 for dw accuracy (the BF16-weight route's det pair, the same dw computation, reaches 0.883 of the per-element budget at T 59392 on B200); up to T 65536 its worst dw element stays within 0.732 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 58881 to 65536 (77 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 72 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_ba7dbf6a07744268705e__58880_65536": {
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
            "above": 58880,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
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
    # sm_100a H4096 bf16 weight: (44032, 65536] above T 44032 for dw accuracy (the FP32-weight route's det pair, the same dw computation, reaches 0.909 of the per-element budget at T 44544 on B200; this route's det pair reaches 0.923 at T 52737 and 0.897 at T 63488 on B200); up to T 65536 its worst dw element stays within 0.774 of the per-element budget at every sampled T from 44033 to 65536 (105 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 70 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_15e157d2ee9080d206a9__44032_65536": {
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
        "kernel": "cake_rmsnorm_train_15e157d2ee9080d206a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_15e157d2ee9080d206a9",
        "role": "bwd_main",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
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
    # sm_100a H4096 f32 weight: (44032, 65536] above T 44032 for dw accuracy (the det pair's worst dw element reaches 0.909 of the per-element budget at T 44544 and 0.891 at T 60416 on B200, and 0.983 at T 63488 in the kernel-tests harness; the BF16-weight route's det pair, the same dw computation, reaches 0.923 at T 52737 on B200); up to T 65536 its worst dw element stays within 0.785 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 44033 to 65536 (132 T, every multiple of 64 rows from 61504 to 65536 among them, worst of 60 inputs of each of three families in both rstd forms at 96 of them) on B200; the twosum form of this kernel runs above T 65536
    "cake_rmsnorm_train_f7573bcb748c0784f3ec__44032_65536": {
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
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
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
    # sm_103a H2048 bf16 weight: (61952, inf] above T 61952 for dw accuracy (both weight dtypes switch at T 61952: the FP32-weight route's det pair, the same dw computation, reaches 0.904 of the per-element budget at T 62464 on GB300); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_8bd89a4c834f80db8a34": {
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
            "above": 61952,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H2048 f32 weight: (61952, inf] above T 61952 for dw accuracy (the det pair's worst dw element reaches 0.904 of the per-element budget at T 62464 on GB300); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_88c6c6b2af1922f4c1a9": {
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
            "above": 61952,
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
    # sm_107a H4096 f32 weight: (64, 256] the column-task form (32 columns per task; twosum: exact per-thread dw sums) on the window of the single kernel it replaces: faster than it at every sampled T of the window on R200 with 212 SMs (1.91x to 2.67x in four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T of the window on R200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on R200 with 216 SMs, 0.0256 against 0.0233 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_0af20ef9ee6868c5569d__64_256": {
        "arches": [
            "sm_107a",
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
        "kernel": "cake_rmsnorm_train_0af20ef9ee6868c5569d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0af20ef9ee6868c5569d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
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
    # sm_100a H512 bf16 weight: (0, 227] the deterministic package's column-task program of this window (16 columns per task; twosum: exact per-thread dw sums): faster than the single kernel of (0, 26] and the pair of (26, 1024] it replaces at every sampled T from 1 to 227 in the balanced small-T sweep on B200 (lane RB, FI main) (their time 1.18x to 1.33x its); the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 26 on B200, 0.0062 against 0.0061 of the per-element budget)
    "cake_rmsnorm_train_5f3e3fccaea795d0a806__0_227": {
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
            "above": 0,
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
    # sm_107a H4096 f32 weight: (0, 64] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.13x to 1.41x the det package's speed and 2.50x to 3.31x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on R200 with 216 SMs, 0.0256 against 0.0233 of the per-element budget)
    "cake_rmsnorm_train_ce1f7d216ecc0a1573b9__0_64_107": {
        "arches": [
            "sm_107a",
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.46x to 1.64x the det package's speed and 2.04x to 2.88x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 14 on B200, 0.0048 against 0.0047 of the per-element budget)
    "cake_rmsnorm_train_38e972a94141a1961c90__0_64": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.40x to 1.68x the det package's speed and 2.49x to 3.06x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on B200, 0.0227 against 0.0203 of the per-element budget)
    "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 f32 weight: (0, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.16x to 1.41x the det package's speed and 1.71x to 3.79x the programs it replaces at every sampled T from 1 to 64 on B200 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on B200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on B200, 0.0134 against 0.0130 of the per-element budget)
    "cake_rmsnorm_train_244b7d841d62d9721c99__0_64": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.31x to 1.59x the det package's speed and 1.44x to 3.77x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 14 on GB300, 0.0048 against 0.0047 of the per-element budget)
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 bf16 weight: (0, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.17x to 1.62x the det package's speed and 1.24x to 4.56x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 12 on GB300, 0.0047 against 0.0045 of the per-element budget)
    "cake_rmsnorm_train_ef052c0918725ffa68d4__0_64_103": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H256 bf16 weight: (0, 1080] for dw accuracy at small T (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at 1080 rows (45 tokens of 24 heads) on GB300, 0.0735 against 0.0728 of the per-element budget)
    "cake_rmsnorm_train_195c4317b51bb4ea1137__0_1080": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_103a H5120 bf16 weight: (0, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.03x to 1.44x the det package's speed and 1.02x to 1.44x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on GB300, 0.0134 against 0.0119 of the per-element budget)
    "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__0_64_103": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.33x to 1.57x the det package's speed and 2.73x to 3.69x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 320 on GB300, 0.0596 against 0.0583 of the per-element budget, and at T 64, 0.0227 against 0.0203)
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (64, 320] for dw accuracy at small T (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 320 on GB300, 0.0596 against 0.0583 of the per-element budget, and at T 64, 0.0227 against 0.0203); twosum: exact per-thread dw sums; T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_963a7ca8d03f78bbe6f9__64_320": {
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
            "above": 64,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H2048 f32 weight: (0, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.23x to 1.57x the det package's speed and 1.21x to 4.32x the programs it replaces at every sampled T from 1 to 64 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 26 on GB300, 0.0152 against 0.0134 of the per-element budget, and at T 42 in the kernel-tests harness, 0.0205 against 0.0189)
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H256 f32 weight: (0, 1080] for dw accuracy at small T (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at 1080 rows (45 tokens of 24 heads) on GB300, 0.0735 against 0.0728 of the per-element budget)
    "cake_rmsnorm_train_8de0131445eed6eeabb7__0_1080": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_103a H4096 f32 weight: (0, 32] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.28x to 1.44x the det package's speed and 1.40x to 4.45x the programs it replaces at every sampled T from 1 to 32 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 15 on GB300, 0.0105 against 0.0092 of the per-element budget)
    "cake_rmsnorm_train_c1b9ee6b9f10766cd537__0_32": {
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
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H5120 f32 weight: (0, 48] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.03x to 1.47x the det package's speed and 1.03x to 5.03x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 22 on GB300, 0.0134 against 0.0130 of the per-element budget)
    "cake_rmsnorm_train_244b7d841d62d9721c99__0_48": {
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
            "above": 0,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 f32 weight: (0, 48] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.03x to 1.53x the det package's speed and 1.04x to 1.53x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 20 on GB300, 0.0147 against 0.0138 of the per-element budget)
    "cake_rmsnorm_train_9b37923697ee8ca52a9e__0_48": {
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
            "above": 0,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (0, 48] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.08x to 1.55x the det package's speed and 1.07x to 1.92x the programs it replaces at every sampled T from 1 to 48 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 6 on GB300, 0.0067 against 0.0063 of the per-element budget)
    "cake_rmsnorm_train_ac9d69f94aabfc68aa40__0_48": {
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
            "above": 0,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.25x to 1.48x the det package's speed and 1.25x to 2.32x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 14 on R200 with 212 SMs, 0.0048 against 0.0047 of the per-element budget)
    "cake_rmsnorm_train_38e972a94141a1961c90__0_64_107": {
        "arches": [
            "sm_107a",
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 bf16 weight: (0, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.14x to 1.52x the det package's speed and 1.14x to 3.00x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 6 on R200 with 212 SMs, 0.0028 against 0.0027 of the per-element budget)
    "cake_rmsnorm_train_ef052c0918725ffa68d4__0_64_107": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H256 bf16 weight: (0, 1056] for dw accuracy at small T (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at 1056 rows (44 tokens of 24 heads) on R200 with 212 SMs, 0.0871 against 0.0657 of the per-element budget)
    "cake_rmsnorm_train_195c4317b51bb4ea1137__0_1056": {
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
        "kernel": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H1536 f32 weight: (64, 256] the column-task form (96 columns per task; twosum: exact per-thread dw sums): the single kernel of (0, 64] exceeded the FP32 eager reference's dw error at T 3 and the pair of (64, 1024] at T 68, 109 and 120 on R200 (torch / fi rstd); the pair keeps (256, 1024]; its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 1 to 1025 on R200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on R200 with 212 SMs, 0.0227 against 0.0203 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e__64_256": {
        "arches": [
            "sm_107a",
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
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 f32 weight: (64, 127] the column-task form (32 columns per task; twosum: exact per-thread dw sums) on the window of the single kernel it replaces: faster than it at every sampled T of the window on R200 with 212 SMs (1.59x to 1.89x in four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T of the window on R200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 127 on R200 with 212 SMs, 0.0339 against 0.0304 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_7a3e5eb75314492eebdb__64_127": {
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
        "kernel": "cake_rmsnorm_train_7a3e5eb75314492eebdb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7a3e5eb75314492eebdb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 127,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H256 f32 weight: (0, 36864] the single kernel for dw accuracy: the pair of (1296, 1536] exceeded the FP32 eager reference's dw error at 20 of its 240 row counts on R200; its worst dw error stays within 0.040 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 1296 to 1537 (every row) on R200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at 1296 rows (54 tokens of 24 heads) on R200 with 212 SMs, 0.0772 against 0.0742 of the per-element budget)
    "cake_rmsnorm_train_8de0131445eed6eeabb7__0_36864": {
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
        "kernel": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H512 f32 weight: (0, 229] the column-task form (16 columns per task; twosum: exact per-thread dw sums) on the window of the single kernel it replaces: faster than it at every sampled T of the window on R200 with 212 SMs (1.12x to 1.21x in four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T of the window on R200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 229 on R200 with 212 SMs, 0.0461 against 0.0384 of the per-element budget)
    "cake_rmsnorm_train_942503b80a21f22cd392__0_229": {
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
            "above": 0,
            "at_most": 229,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (64, 2304] the column-task form (twosum: exact per-thread dw sums); from T 20 for dw accuracy: the pair of (20, 28672] exceeded the FP32 eager reference's dw error at 18 sampled T from 4544 to 19264 on R200 with 212 SMs (the deterministic package runs the same pair there); its worst dw error stays at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges on R200, worst of 60 inputs of each of three families in both rstd forms; faster than the single kernel of (2304, 19456] up to T 2304; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 20 on R200 with 212 SMs, 0.0147 against 0.0138 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_44c5b870a7dcbb8de924__64_2304": {
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
            "above": 64,
            "at_most": 2304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (64, 96] the column-task form (twosum: exact per-thread dw sums): the pair of (3, 448] exceeded the FP32 eager reference's dw error at T 8 on R200 (0.0083 against 0.0075 of the per-element budget, torch rstd); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T from 1 to 64 and every 8 rows to 449 on R200, worst of 60 inputs of each of three families in both rstd forms; level with or faster than that pair up to T 96 on R200, the pair keeps (96, 448]; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 3 on R200 with 212 SMs, 0.0041 against 0.0034 of the per-element budget); T 1 to 64 runs the fixed-claim column-task form
    "cake_rmsnorm_train_c8bc10e76c624f5f4534__64_96": {
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
        "kernel": "cake_rmsnorm_train_c8bc10e76c624f5f4534",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c8bc10e76c624f5f4534",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 64,
            "at_most": 96,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H4096 bf16 weight: (0, 64] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.14x to 1.40x the det package's speed and 1.14x to 3.27x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 6 on R200 with 212 SMs, 0.0031 against 0.0030 of the per-element budget)
    "cake_rmsnorm_train_231e01eca0459c2c0bf5__0_64_107": {
        "arches": [
            "sm_107a",
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 16 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.25x to 1.44x the det package's speed and 2.01x to 2.25x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 64 on R200 with 212 SMs, 0.0227 against 0.0203 of the per-element budget)
    "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64_107": {
        "arches": [
            "sm_107a",
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 f32 weight: (0, 64] the fixed-claim column-task form (64 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.09x to 1.39x the det package's speed and 2.26x to 2.85x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 127 on R200 with 212 SMs, 0.0339 against 0.0304 of the per-element budget)
    "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64_107": {
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
        "kernel": "cake_rmsnorm_train_382e45788ace3ff40ba1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_382e45788ace3ff40ba1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.18x to 1.57x the det package's speed and 1.18x to 4.02x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 20 on R200 with 212 SMs, 0.0147 against 0.0138 of the per-element budget)
    "cake_rmsnorm_train_9b37923697ee8ca52a9e__0_64_107": {
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
        "kernel": "cake_rmsnorm_train_9b37923697ee8ca52a9e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9b37923697ee8ca52a9e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.21x to 1.48x the det package's speed and 1.21x to 1.47x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 3 on R200 with 212 SMs, 0.0041 against 0.0034 of the per-element budget)
    "cake_rmsnorm_train_ac9d69f94aabfc68aa40__0_64_107": {
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
        "kernel": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ac9d69f94aabfc68aa40",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 bf16 weight: (0, 32] the fixed-claim column-task form (128 column tasks of 32 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.27x to 1.47x the det package's speed and 1.42x to 4.61x the programs it replaces at every sampled T from 1 to 32 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 3 on GB300, by less than 0.0001 of the per-element budget)
    "cake_rmsnorm_train_3dbab2323fa42435a72d__0_32": {
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
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (0, 32] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.20x to 1.51x the det package's speed and 1.19x to 1.95x the programs it replaces at every sampled T from 1 to 32 on GB300 (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on GB300, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 3 on GB300, by less than 0.0001 of the per-element budget)
    "cake_rmsnorm_train_d70f2562a592151ec4f9__0_32": {
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
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.19x to 1.60x the det package's speed and 1.20x to 4.05x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 3 on R200 with 212 SMs, by less than 0.0001 of the per-element budget)
    "cake_rmsnorm_train_d70f2562a592151ec4f9__0_64_107": {
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
        "kernel": "cake_rmsnorm_train_d70f2562a592151ec4f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d70f2562a592151ec4f9",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (0, 64] the fixed-claim column-task form (96 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.23x to 1.52x the det package's speed and 1.25x to 1.56x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 3 on R200 with 212 SMs, by less than 0.0001 of the per-element budget)
    "cake_rmsnorm_train_7b0f5a6a8769473c322e__0_64_107": {
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
        "kernel": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7b0f5a6a8769473c322e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H512 bf16 weight: (0, 26] for dw accuracy at small T (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 26 on GB300, 0.0062 against 0.0061 of the per-element budget)
    "cake_rmsnorm_train_dd5c218dacb8aa174790": {
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
        "kernel": "cake_rmsnorm_train_dd5c218dacb8aa174790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dd5c218dacb8aa174790",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 26,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 8192,
        },
    },
    # sm_107a H512 bf16 weight: (0, 77] the column-task form (16 columns per task; twosum: exact per-thread dw sums) on the window of the single kernel it replaces: faster than it at every sampled T of the window on R200 with 212 SMs (1.05x to 1.21x in four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every sampled T of the window on R200, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (the det pair's median dw error over the sampled inputs exceeds the FP32 eager reference's at T 77 on R200 with 212 SMs, 0.0103 against 0.0100 of the per-element budget)
    "cake_rmsnorm_train_5f3e3fccaea795d0a806__0_77": {
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
            "above": 0,
            "at_most": 77,
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
    # sm_107a H5120 f32 weight: (49152, inf] above T 49152, where the det pair stops; kblock pipe single (FP32 sums of 8-row blocks kept as FP32 TwoSum pairs in registers): worst dw element 0.719 of the budget at 265 sampled T from 49153 to 131072 on R200 (212 SMs, 3 x 60 inputs per T, kernel and torch rstd), at most the FP32 eager reference's, 1.66 to 1.78 times the FP64-row single's speed in the FI bench (R200, 4 sessions)
    "cake_rmsnorm_train_6cedccdf3a58f23f260f": {
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
    # sm_100a H512 f32 weight: (0, 227] the deterministic package's column-task program of this window (16 columns per task; twosum: exact per-thread dw sums): faster than the single kernel of (0, 292] it replaces at every sampled T from 1 to 227 in the balanced small-T sweep on B200 (lane RB, FI main) (its time 1.30x to 1.41x); the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 288 on B200, 0.0521 against 0.0409 of the per-element budget, and the FP64-accumulating single kernel this route runs above T 960 exceeds it at T 4, 0.0033 against 0.0032; with the kernel-tests harness inputs included the det pair's worst dw error exceeds it at T 292, 0.0558 against 0.0554)
    "cake_rmsnorm_train_942503b80a21f22cd392__0_227": {
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
            "above": 0,
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
    # sm_107a H5120 bf16 weight: (0, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.29x to 1.71x the det package's speed and 1.29x to 3.56x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 15 on R200 with 212 SMs, 0.0120 against 0.0114 of the per-element budget, and at T 18 in the kernel-tests harness, 0.0136 against 0.0134)
    "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__0_64_107": {
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
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (0, 64] the fixed-claim column-task form (80 column tasks of 64 columns; each CTA takes its rows and column tasks by its index, no claim tickets; twosum: exact per-thread dw sums; dx and dw bitwise its ticket twin's): 1.26x to 1.71x the det package's speed and 1.26x to 3.50x the programs it replaces at every sampled T from 1 to 64 on R200 with 212 SMs (CUDA-graph span, four sessions); its worst dw error stays within 0.0006 of the per-element budget and at or below the FP32 eager reference's at every T from 1 to 64 on R200 with 212 SMs, worst of 60 inputs of each of three families in both rstd forms; the det pair stays out of this window for dw accuracy (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 13 on R200 with 212 SMs, 0.0127 against 0.0113 of the per-element budget)
    "cake_rmsnorm_train_244b7d841d62d9721c99__0_64_107": {
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
        "kernel": "cake_rmsnorm_train_244b7d841d62d9721c99",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_244b7d841d62d9721c99",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H512 f32 weight: (0, 292] for dw accuracy at small T (over 60 inputs of each of three families the det pair's worst dw error exceeds the FP32 eager reference's at T 288 on GB300, 0.0521 against 0.0409 of the per-element budget, and an FP64-accumulating single kernel of the same tiling, not shipped, exceeds it at T 4, 0.0033 against 0.0032; with the kernel-tests harness inputs included the det pair's worst dw error exceeds it at T 292, 0.0558 against 0.0554); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_4724ad6abf88e40c2151": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_4724ad6abf88e40c2151",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4724ad6abf88e40c2151",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 292,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 8192,
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
    "cake_rmsnorm_train_7dc29629ff21625607f8__64_1024_103": {
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
            "above": 64,
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
    "cake_rmsnorm_train_b457c4d134e44476dc6f__32768_36864": {
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
            "at_most": 36864,
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
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__64_1024_103": {
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
            "above": 64,
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
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_36864": {
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
            "at_most": 36864,
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
    "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_13608": {
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
            "at_most": 13608,
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
    "cake_rmsnorm_train_d0960be986e58f449f50__11264_13608": {
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
            "at_most": 13608,
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
    "cake_rmsnorm_train_09601d42628de24f6f87__26_1024": {
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
        "kernel": "cake_rmsnorm_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_09601d42628de24f6f87",
        "role": "bwd_small",
        "select_rule": {
            "above": 26,
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
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__26_1024": {
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
            "above": 26,
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
    "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_5632": {
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
            "at_most": 5632,
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
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_5632": {
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
            "at_most": 5632,
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
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__320_1024": {
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
            "above": 320,
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
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__320_1024": {
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
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__64_3072": {
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
            "above": 64,
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
    "cake_rmsnorm_train_880768a0a264e04dbfb5__64_3072": {
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
            "above": 64,
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
    "cake_rmsnorm_train_07f9f2fbea7163639c27__64_3072": {
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
            "above": 64,
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
    "cake_rmsnorm_train_50044ee3bd1ca54bd821__32_256": {
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
            "above": 32,
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
    "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_50176": {
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
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__32_256": {
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
            "above": 32,
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
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_50176": {
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
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3fd77c8ce97e8a523abb__32_256": {
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
            "above": 32,
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
    "cake_rmsnorm_train_626aed743358c27513f1__4096_9216": {
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
            "at_most": 9216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_9216": {
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
            "at_most": 9216,
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
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__64_5120": {
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
    "cake_rmsnorm_train_79811a76205d7d7dd197__48_5120": {
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
            "above": 48,
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
    "cake_rmsnorm_train_c5831b29d17867842e5f__64_5120": {
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
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__48_5120": {
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
            "above": 48,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__32_4096": {
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
            "above": 32,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_322574af704b51aef0bb__4096_49152": {
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
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__32_4096": {
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
            "above": 32,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__448_4096": {
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
            "above": 448,
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
    "cake_rmsnorm_train_41b80b6af97974feb181__48_320": {
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
            "above": 48,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_41b80b6af97974feb181__448_4096": {
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
            "above": 448,
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
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__48_320": {
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
            "above": 48,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__56_304": {
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
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 56,
            "at_most": 304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__1920_2816": {
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
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1920,
            "at_most": 2816,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__4608_5632": {
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
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 4608,
            "at_most": 5632,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2e8d2128a33361628509__56_304": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 56,
            "at_most": 304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2e8d2128a33361628509__1920_2816": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1920,
            "at_most": 2816,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2e8d2128a33361628509__4608_5632": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4608,
            "at_most": 5632,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56f988dde54db37f09d9__48_320": {
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
        "kernel": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 48,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_56f988dde54db37f09d9__1824_3584": {
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
        "kernel": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1824,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_56f988dde54db37f09d9__4608_5632": {
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
        "kernel": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 4608,
            "at_most": 5632,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2e8d2128a33361628509__48_320": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 48,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2e8d2128a33361628509__1824_3584": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1824,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7dc29629ff21625607f8__64_1024_107": {
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
            "above": 64,
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
    "cake_rmsnorm_train_56d26996239835327501__16384_53248": {
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
            "at_most": 53248,
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
    "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_53248": {
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
            "at_most": 53248,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__256_1024": {
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
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_53248": {
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
            "at_most": 53248,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_628248febbdfe17c3325__256_1024": {
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
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_628248febbdfe17c3325__64_1024": {
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
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_84efa2af7907877dfb56__64_256": {
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
        "kernel": "cake_rmsnorm_train_84efa2af7907877dfb56",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_84efa2af7907877dfb56",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
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
    "cake_rmsnorm_train_520453292492bd2725be__64_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_520453292492bd2725be",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_520453292492bd2725be",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_6b13aa49e0c0414d9041__127_256": {
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
        "kernel": "cake_rmsnorm_train_6b13aa49e0c0414d9041",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6b13aa49e0c0414d9041",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 127,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
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
    "cake_rmsnorm_train_520453292492bd2725be__127_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_520453292492bd2725be",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_520453292492bd2725be",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 127,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5450d39836b26cfd3256__1056_3584": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_5450d39836b26cfd3256",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5450d39836b26cfd3256",
        "role": "bwd_main",
        "select_rule": {
            "above": 1056,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_312bbe084eac69d858c4__1056_3584": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1056,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0da5dd4b80ad9b5426ce__64_256": {
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
        "kernel": "cake_rmsnorm_train_0da5dd4b80ad9b5426ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da5dd4b80ad9b5426ce",
        "role": "bwd_small",
        "select_rule": {
            "above": 64,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
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
    "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_36864": {
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
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_69bf6e104bb48b6b74bf__64_256": {
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
            "above": 64,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_10e24516314c244771f4__4096_36864": {
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
            "at_most": 36864,
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
    "cake_rmsnorm_train_3feed5026e1156d2fba2__77_1024": {
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
            "above": 77,
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
    "cake_rmsnorm_train_f94861c36a14eb75e39c__77_1024": {
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
            "above": 77,
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
    "cake_rmsnorm_train_b841d64bbcaefff5ce7b__229_1024": {
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
            "above": 229,
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
    "cake_rmsnorm_train_f94861c36a14eb75e39c__229_1024": {
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
            "above": 229,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__64_4096": {
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
            "above": 64,
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
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__64_4096": {
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
            "above": 64,
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
    # sm_107a H5120 f32 weight: (4096, 19968] a row single kernel (it ran (49152, inf) before the kblock kernel there) for dw accuracy: the pair of the window exceeded the FP32 eager reference's dw error at T 13024, 17472, 17728 and 19520 and at T 18304 (torch rstd) on R200 with 212 SMs (the deterministic package runs the same pair there); its worst dw error stays at or below the FP32 eager reference's at every multiple of 64 rows of the window and both edges on R200, worst of 60 inputs of each of three families in both rstd forms
    "cake_rmsnorm_train_dd795e849940a49d0d29__4096_19968": {
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
        "kernel": "cake_rmsnorm_train_dd795e849940a49d0d29",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dd795e849940a49d0d29",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    "cake_rmsnorm_train_cd612de56c0985bb4caa__64_4096": {
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
            "above": 64,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7b9e10b21b75d4615135__19968_49152": {
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
        "kernel": "cake_rmsnorm_train_7b9e10b21b75d4615135",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7b9e10b21b75d4615135",
        "role": "bwd_main_w32",
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
    "cake_rmsnorm_train_5d292172df772adb17ae__64_19456": {
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
            "above": 64,
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
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__64_19456": {
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
            "above": 64,
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
    "cake_rmsnorm_train_0dcdc563be377247044e__64_448": {
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
            "above": 64,
            "at_most": 448,
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
    "cake_rmsnorm_train_60bfcb24560bbae612c3__96_448": {
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
            "above": 96,
            "at_most": 448,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_60bfcb24560bbae612c3__64_448": {
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
            "above": 64,
            "at_most": 448,
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
    "cake_rmsnorm_train_56f988dde54db37f09d9__96_448": {
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
            "above": 96,
            "at_most": 448,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
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
            "cake_rmsnorm_train_195c4317b51bb4ea1137",
            "cake_rmsnorm_train_172a47a15d4c1a76c061__20480_92160",
            "cake_rmsnorm_train_083ea823c691e0631ad3__92160_524288",
            "cake_rmsnorm_train_172a47a15d4c1a76c061",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__92160_524288",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_8de0131445eed6eeabb7",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__20480_92160",
            "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__92160_524288",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__92160_524288",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_5f3e3fccaea795d0a806__0_227",
            "cake_rmsnorm_train_09601d42628de24f6f87__227_1024",
            "cake_rmsnorm_train_123db0b2f7fb38cfab03__1024_1280",
            "cake_rmsnorm_train_d0bde38aa443d42a5f04",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_train_fef03ce50a792c8704bd__1024_1280",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_942503b80a21f22cd392__0_227",
            "cake_rmsnorm_train_aa7c5a5407cba063b9d5__227_292",
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__292_960",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__292_960",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_38e972a94141a1961c90__0_64",
            "cake_rmsnorm_train_dc478ac1195882aedee6__64_384",
            "cake_rmsnorm_train_7dc29629ff21625607f8__384_1024",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_21c8bb55c8d672b63952",
            "cake_rmsnorm_train_b457c4d134e44476dc6f__32768_36864",
            "cake_rmsnorm_train_ae610afdf77ff6714a5b__36864_78784",
            "cake_rmsnorm_train_66238d219de588fca483",
            "cake_rmsnorm_train_b42efd6bf06148d75151__131072_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_36864",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64",
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
            "cake_rmsnorm_train_ef052c0918725ffa68d4__0_64",
            "cake_rmsnorm_train_64e6aa82bb2c6d4ea978__64_336",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__336_3072",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__4096_7168",
            "cake_rmsnorm_train_0fa4f255caf709d09f66",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_13608",
            "cake_rmsnorm_train_0fa4f255caf709d09f66__13608_45056",
            "cake_rmsnorm_train_0fa4f255caf709d09f66__45056_58880",
            "cake_rmsnorm_train_0fa4f255caf709d09f66__58880_65536",
            "cake_rmsnorm_train_8b8cf4f5b82e77554908",
            "cake_rmsnorm_train_83f905ffd00c0ef641e2__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_7168",
            "cake_rmsnorm_train_d0960be986e58f449f50__11264_13608",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64",
            "cake_rmsnorm_train_17493054fed5409b6df9__64_336",
            "cake_rmsnorm_train_17493054fed5409b6df9__336_1024",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__1024_4096",
            "cake_rmsnorm_train_b187e5989302def22b2d__4096_15403",
            "cake_rmsnorm_train_ba7dbf6a07744268705e",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__45056_58880",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__58880_65536",
            "cake_rmsnorm_train_273d3f9b357669a821e2",
            "cake_rmsnorm_train_9f775c72c1335f4b08ac__131072_inf",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_15403",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_231e01eca0459c2c0bf5__0_64",
            "cake_rmsnorm_train_4822793ea0c954c11af6__64_1152",
            "cake_rmsnorm_train_9a830956447d4e304914__1152_4096",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_4608",
            "cake_rmsnorm_train_15e157d2ee9080d206a9",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_44032",
            "cake_rmsnorm_train_15e157d2ee9080d206a9__44032_65536",
            "cake_rmsnorm_train_fd7ae1d7547fca6156a0",
            "cake_rmsnorm_train_c67fc8b9d1b1edb8d7da__131072_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1152_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_4608",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_ce1f7d216ecc0a1573b9__0_64",
            "cake_rmsnorm_train_8616070567821eaa3e35__64_1792",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec__1792_3584",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec",
            "cake_rmsnorm_train_626aed743358c27513f1__36864_44032",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec__44032_65536",
            "cake_rmsnorm_train_6e65495154249f51e74b",
            "cake_rmsnorm_train_9ebd0232954f1b1842f7__131072_inf",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__0_64",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__64_896",
            "cake_rmsnorm_train_b482c84dbffcd75a4a4c__896_6144",
            "cake_rmsnorm_train_c145ef195d7aeb04f242__6144_56256",
            "cake_rmsnorm_train_56634f53e6456b34f76f__56256_131072",
            "cake_rmsnorm_train_ade9f6a4f10b069ba8a3__131072_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__896_6144",
            "cake_rmsnorm_train_c5831b29d17867842e5f__6144_56256",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_244b7d841d62d9721c99__0_64",
            "cake_rmsnorm_train_68f6e584c687c30c0084__64_1792",
            "cake_rmsnorm_train_de8f6091077dc7ac5947__1792_6144",
            "cake_rmsnorm_train_9479b1298a1b19d4d996__6144_56256",
            "cake_rmsnorm_train_6cedccdf3a58f23f260f__56256_131072",
            "cake_rmsnorm_train_fc2a52a2c9bf8c5be9af__131072_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__6144_56256",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_d70f2562a592151ec4f9__0_64",
            "cake_rmsnorm_train_44508df7dab2ab688f85__64_1152",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096",
            "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_5632",
            "cake_rmsnorm_train_9433061da18b2db4299d__5632_65536",
            "cake_rmsnorm_train_995b2229fe43c772b2c7",
            "cake_rmsnorm_train_10a41132a42d594ce6dd__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_5632",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_7b0f5a6a8769473c322e__0_64",
            "cake_rmsnorm_train_864e84f3d874f850c091__64_1408",
            "cake_rmsnorm_train_3fbfdd750cbeb47032b5__1408_65536",
            "cake_rmsnorm_train_7c216db7f9e0c6fd9c4d",
            "cake_rmsnorm_train_1609f7969c50fba7f2c7__131072_inf",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_9b37923697ee8ca52a9e__0_64",
            "cake_rmsnorm_train_b3e6531e46667901ab91__64_1168",
            "cake_rmsnorm_train_41b80b6af97974feb181__1168_3584",
            "cake_rmsnorm_train_a562656cef5bf5dda160__3584_65536",
            "cake_rmsnorm_train_d5a17016b7ff2c6e3ba9",
            "cake_rmsnorm_train_45cf60e3cd8603c2abeb__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1168_3584",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_ac9d69f94aabfc68aa40__0_64",
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
            "cake_rmsnorm_train_195c4317b51bb4ea1137__0_1080",
            "cake_rmsnorm_train_af005462cd6a301f31c4__1080_5120",
            "cake_rmsnorm_train_a385ae1681a2c078344e__5120_12288",
            "cake_rmsnorm_train_af005462cd6a301f31c4__12288_40960",
            "cake_rmsnorm_train_a385ae1681a2c078344e__40960_97280",
            "cake_rmsnorm_train_083ea823c691e0631ad3__97280_720896",
            "cake_rmsnorm_train_0291949ba535c646c3a8__720896_1572864",
            "cake_rmsnorm_train_c18297d75c0d424978fd",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_5120",
            "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__40960_97280",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__97280_720896",
            "cake_rmsnorm_train_0ac243f8385c32960e14__720896_1572864",
            "cake_rmsnorm_train_e73321cec89d33336ea1",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_8de0131445eed6eeabb7__0_1080",
            "cake_rmsnorm_train_b7d9577ff884de98ea06__1080_5120",
            "cake_rmsnorm_train_420788e663a2c3a810f6__5120_12288",
            "cake_rmsnorm_train_b7d9577ff884de98ea06__12288_40960",
            "cake_rmsnorm_train_420788e663a2c3a810f6__40960_97280",
            "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b__97280_720896",
            "cake_rmsnorm_train_89505762fa1cbf33cb8f__720896_1572864",
            "cake_rmsnorm_train_f788ef57163552825d27",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_5120",
            "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__40960_97280",
            "cake_rmsnorm_train_b4b63d5b5028c7a60333__97280_720896",
            "cake_rmsnorm_train_0ac243f8385c32960e14__720896_1572864",
            "cake_rmsnorm_train_e73321cec89d33336ea1",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_dd5c218dacb8aa174790",
            "cake_rmsnorm_train_09601d42628de24f6f87__26_1024",
            "cake_rmsnorm_train_8e7f4d811e55a351a22e",
            "cake_rmsnorm_train_274b0055aace183f2367",
            "cake_rmsnorm_train_cfaae3cec98f8501832b",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__26_1024",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_train_122721bd6b4650866553__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_4724ad6abf88e40c2151",
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__292_960",
            "cake_rmsnorm_train_56ee8e66479625fd078a__960_32768",
            "cake_rmsnorm_train_a546f410ec8cf03ef20d",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__292_960",
            "cake_rmsnorm_train_122721bd6b4650866553__960_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_38e972a94141a1961c90__0_64_103",
            "cake_rmsnorm_train_7dc29629ff21625607f8__64_1024_103",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_1df7db17277d363b5168",
            "cake_rmsnorm_train_4df08ce51fb01ba1193f",
            "cake_rmsnorm_train_f522f5b8d4343d96ca04__15566_46592",
            "cake_rmsnorm_train_327bfeb4ab8fefc2731d",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__64_1024_103",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_train_1863320f3158d35eead6__15566_46592",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64_103",
            "cake_rmsnorm_train_963a7ca8d03f78bbe6f9__64_320",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__320_1024",
            "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
            "cake_rmsnorm_train_a99e2b380514c4d86d60",
            "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2",
            "cake_rmsnorm_train_109f19bb177c8f20cadc__15566_37888",
            "cake_rmsnorm_train_963a7ca8d03f78bbe6f9",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__320_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_train_1863320f3158d35eead6__15566_37888",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_ef052c0918725ffa68d4__0_64_103",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__64_3072",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__large",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768",
            "cake_rmsnorm_train_302f64db6abe79e5259e__32768_58496",
            "cake_rmsnorm_train_018f0cd723106f709f11__58496_61952",
            "cake_rmsnorm_train_8bd89a4c834f80db8a34",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__64_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496",
            "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64_103",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__64_3072",
            "cake_rmsnorm_train_d96a14fe8882bbd9ec2e",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__large",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__16384_32768",
            "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_58496",
            "cake_rmsnorm_train_ca8ca4674a3dbd89ad16__58496_61952",
            "cake_rmsnorm_train_88c6c6b2af1922f4c1a9",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__64_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496",
            "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_3dbab2323fa42435a72d__0_32",
            "cake_rmsnorm_train_50044ee3bd1ca54bd821__32_256",
            "cake_rmsnorm_train_ad2bb26ea5916baef963",
            "cake_rmsnorm_train_24643edc7e1143da24e0",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_50176",
            "cake_rmsnorm_train_8250eb924d925c0db0d9",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__32_256",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_50176",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_c1b9ee6b9f10766cd537__0_32",
            "cake_rmsnorm_train_3fd77c8ce97e8a523abb__32_256",
            "cake_rmsnorm_train_16545056022785f2a7d8",
            "cake_rmsnorm_train_78c26a53e83c9803bcbf",
            "cake_rmsnorm_train_626aed743358c27513f1__4096_9216",
            "cake_rmsnorm_train_2911faf7a786144495a6",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__32_256",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_9216",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__0_64_103",
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__64_5120",
            "cake_rmsnorm_train_7ffb443a63a92ab6b0e0__5120_61952",
            "cake_rmsnorm_train_b523a764b029e5ca1aaf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__64_5120",
            "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_244b7d841d62d9721c99__0_48",
            "cake_rmsnorm_train_79811a76205d7d7dd197__48_5120",
            "cake_rmsnorm_train_f9f7d4cd75ef91cc8374__5120_61952",
            "cake_rmsnorm_train_90598c14b4e396f5c8b0",
            "cake_rmsnorm_train_c5831b29d17867842e5f__48_5120",
            "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_d70f2562a592151ec4f9__0_32",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__32_4096",
            "cake_rmsnorm_train_322574af704b51aef0bb__4096_49152",
            "cake_rmsnorm_train_582b2a20f9be648af8ec",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__32_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_7b0f5a6a8769473c322e__0_48",
            "cake_rmsnorm_train_864e84f3d874f850c091__48_56",
            "cake_rmsnorm_train_0dcdc563be377247044e__56_304",
            "cake_rmsnorm_train_864e84f3d874f850c091__304_1920",
            "cake_rmsnorm_train_0dcdc563be377247044e__1920_2816",
            "cake_rmsnorm_train_115dcc719a6de4b079f1",
            "cake_rmsnorm_train_0dcdc563be377247044e__4608_5632",
            "cake_rmsnorm_train_115dcc719a6de4b079f1__5632_32768",
            "cake_rmsnorm_train_52831494c1b62db758fd",
            "cake_rmsnorm_train_2e8d2128a33361628509__56_304",
            "cake_rmsnorm_train_2e8d2128a33361628509__1920_2816",
            "cake_rmsnorm_train_2e8d2128a33361628509__4608_5632",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_9b37923697ee8ca52a9e__0_48",
            "cake_rmsnorm_train_41b80b6af97974feb181__48_320",
            "cake_rmsnorm_train_b3e6531e46667901ab91__320_448",
            "cake_rmsnorm_train_41b80b6af97974feb181__448_4096",
            "cake_rmsnorm_train_9a8c0396edd85d4d545e__4096_49152",
            "cake_rmsnorm_train_4f9a76e68671f213ab7c",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__48_320",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__448_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_ac9d69f94aabfc68aa40__0_48",
            "cake_rmsnorm_train_56f988dde54db37f09d9__48_320",
            "cake_rmsnorm_train_71d36617003ab7742a77__320_1824",
            "cake_rmsnorm_train_56f988dde54db37f09d9__1824_3584",
            "cake_rmsnorm_train_3c390339fa8c82311a0a",
            "cake_rmsnorm_train_56f988dde54db37f09d9__4608_5632",
            "cake_rmsnorm_train_3c390339fa8c82311a0a__5632_15872",
            "cake_rmsnorm_train_56c46a201c168050aa60",
            "cake_rmsnorm_train_2e8d2128a33361628509__48_320",
            "cake_rmsnorm_train_2e8d2128a33361628509__1824_3584",
            "cake_rmsnorm_train_2e8d2128a33361628509__4608_5632",
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
            "cake_rmsnorm_train_195c4317b51bb4ea1137__0_1056",
            "cake_rmsnorm_train_5450d39836b26cfd3256__1056_3584",
            "cake_rmsnorm_train_195c4317b51bb4ea1137__3584_36864",
            "cake_rmsnorm_train_172a47a15d4c1a76c061__36864_inf",
            "cake_rmsnorm_train_312bbe084eac69d858c4__1056_3584",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_8de0131445eed6eeabb7__0_36864",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__36864_inf",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_5f3e3fccaea795d0a806__0_77",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__77_1024",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096",
            "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
            "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3__16384_49152",
            "cake_rmsnorm_train_30501ccec48cc058333a__49152_inf",
            "cake_rmsnorm_train_f94861c36a14eb75e39c__77_1024",
            "cake_rmsnorm_train_d75459d901441384398b",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_942503b80a21f22cd392__0_229",
            "cake_rmsnorm_train_b841d64bbcaefff5ce7b__229_1024",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22__1024_4096",
            "cake_rmsnorm_train_383f709cd5c4fadd9436",
            "cake_rmsnorm_train_03efa681eb38f59b00c2__16384_49152",
            "cake_rmsnorm_train_26fbe8e6f07a658e26f6__49152_inf",
            "cake_rmsnorm_train_f94861c36a14eb75e39c__229_1024",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_38e972a94141a1961c90__0_64_107",
            "cake_rmsnorm_train_7dc29629ff21625607f8__64_1024_107",
            "cake_rmsnorm_train_6a428da2227972ae18ae",
            "cake_rmsnorm_train_56d26996239835327501__16384_53248",
            "cake_rmsnorm_train_3f75f4a81f2fb8269389__53248_65536",
            "cake_rmsnorm_train_dca25ace64c075a81625__65536_inf",
            "cake_rmsnorm_train_628248febbdfe17c3325__64_1024",
            "cake_rmsnorm_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_53248",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_d5a621af7a24e403b5f3__0_64_107",
            "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e__64_256",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__256_1024",
            "cake_rmsnorm_train_7c0dd63f0b011049603e__1024_16384",
            "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_53248",
            "cake_rmsnorm_train_7c0dd63f0b011049603e__53248_65536",
            "cake_rmsnorm_train_ccd94d87e9036a0b2faf__65536_inf",
            "cake_rmsnorm_train_628248febbdfe17c3325__256_1024",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_53248",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_ef052c0918725ffa68d4__0_64_107",
            "cake_rmsnorm_train_84efa2af7907877dfb56__64_256",
            "cake_rmsnorm_train_845cdcd8145ada631461__256_1024",
            "cake_rmsnorm_train_81c3d8ab0a482e3af043",
            "cake_rmsnorm_train_845cdcd8145ada631461__16384_50176",
            "cake_rmsnorm_train_1a3df5ac020055477581",
            "cake_rmsnorm_train_520453292492bd2725be__64_256",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50176",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_382e45788ace3ff40ba1__0_64_107",
            "cake_rmsnorm_train_7a3e5eb75314492eebdb__64_127",
            "cake_rmsnorm_train_6b13aa49e0c0414d9041__127_256",
            "cake_rmsnorm_train_838a9a4054731f7942af__256_1024",
            "cake_rmsnorm_train_810989bdea464df479ce__1024_16384",
            "cake_rmsnorm_train_838a9a4054731f7942af__16384_50176",
            "cake_rmsnorm_train_810989bdea464df479ce",
            "cake_rmsnorm_train_520453292492bd2725be__127_256",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50176",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_231e01eca0459c2c0bf5__0_64_107",
            "cake_rmsnorm_train_0da5dd4b80ad9b5426ce__64_256",
            "cake_rmsnorm_train_a647fccc3d9fa9134733",
            "cake_rmsnorm_train_d41a514f5210e9adc39f",
            "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_36864",
            "cake_rmsnorm_train_f11a86167db4f8e9a5da__36864_65536",
            "cake_rmsnorm_train_5e9d7baa36814bb7aaf6__65536_inf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__64_256",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_36864",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_ce1f7d216ecc0a1573b9__0_64_107",
            "cake_rmsnorm_train_0af20ef9ee6868c5569d__64_256",
            "cake_rmsnorm_train_dc439f8335327e482025",
            "cake_rmsnorm_train_b08633208e60b8ce94fd",
            "cake_rmsnorm_train_0f1859b9fe9e47668a1c__4096_24064",
            "cake_rmsnorm_train_a1f537fc94bde175bb3c__24064_65536",
            "cake_rmsnorm_train_23cee954b648495ceef1__65536_inf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_24064",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c__0_64_107",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__64_4096",
            "cake_rmsnorm_train_6816fb80e97380f1bcbc",
            "cake_rmsnorm_train_b8b0913b40f035f43b76__19968_49152",
            "cake_rmsnorm_train_56634f53e6456b34f76f",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__64_4096",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_244b7d841d62d9721c99__0_64_107",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__64_4096",
            "cake_rmsnorm_train_dd795e849940a49d0d29__4096_19968",
            "cake_rmsnorm_train_7b9e10b21b75d4615135__19968_49152",
            "cake_rmsnorm_train_6cedccdf3a58f23f260f",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__64_4096",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_d70f2562a592151ec4f9__0_64_107",
            "cake_rmsnorm_train_5d292172df772adb17ae__64_19456",
            "cake_rmsnorm_train_19e18371f9243bd1a91a__19456_30720",
            "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82__30720_inf",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__64_19456",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_30720",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_7b0f5a6a8769473c322e__0_64_107",
            "cake_rmsnorm_train_0dcdc563be377247044e__64_448",
            "cake_rmsnorm_train_864e84f3d874f850c091__448_3584",
            "cake_rmsnorm_train_0dcdc563be377247044e__3584_7168",
            "cake_rmsnorm_train_c07ab5f16934c0cc8421__7168_inf",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__64_448",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_9b37923697ee8ca52a9e__0_64_107",
            "cake_rmsnorm_train_44c5b870a7dcbb8de924__64_2304",
            "cake_rmsnorm_train_e77387f14390ba486055__2304_19456",
            "cake_rmsnorm_train_b83f8b3d546630a60675__19456_28672",
            "cake_rmsnorm_train_25f9b466fa418dd2afc4__28672_inf",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_28672",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_ac9d69f94aabfc68aa40__0_64_107",
            "cake_rmsnorm_train_c8bc10e76c624f5f4534__64_96",
            "cake_rmsnorm_train_56f988dde54db37f09d9__96_448",
            "cake_rmsnorm_train_71d36617003ab7742a77__448_3584",
            "cake_rmsnorm_train_56f988dde54db37f09d9__3584_7168",
            "cake_rmsnorm_train_e655055ead76f8df4d24__7168_inf",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__96_448",
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
    """The weight dtypes (tokens of ``WEIGHT_DTYPES``) the route ships for on ``arch``."""

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
# (``PAIR_ROWS_OFFSET``).  Every offset is the package's, not the call's, so one
# buffer, zero when first used, serves calls of any token count, window, hidden
# size, residual flag and weight dtype with no host write between them, and the
# two forms never share a counter word.
COUNTERS_BYTES = 1024
SINGLE_COUNTERS_OFFSET = COUNTERS_BYTES
SINGLE_COUNTERS_BYTES = 256
SUMS_OFFSET = SINGLE_COUNTERS_OFFSET + SINGLE_COUNTERS_BYTES
# The workspace rule kinds of the two backward forms: the main kernel of a
# deterministic pair, the single kernel.
WORKSPACE_KINDS = ("chunk_partials", "claim_sum_f64")


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


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``rows_per_cta`` modules launch ``ceil(rows / rows_per_cta)`` CTAs, capped
    at ``grid_limit`` when the rule records a positive limit, else at
    ``grid_per_sm * SM count`` of the device when that is positive (persistent
    grid-stride row loop in the kernel).  ``fixed`` modules launch exactly
    ``ctas`` CTAs (the dw reduce kernel: one CTA per column slab).
    ``claimed_chunks`` modules are the main backward kernel: ``slots =
    ctas_per_sm * SM count`` persistent CTAs claim the chunks of the table of
    :func:`chunk_rounds` (``rows_per_issue`` rows per pipeline step of the
    kernel, under the taper its record spells: :func:`chunk_taper`), so the
    grid is ``min(slots, n_chunks)``; a column-task kernel's rule spells its
    ``col_tasks`` (``H / col`` column tasks, tickets beside the chunks), so its
    grid is ``min(slots, n_chunks + col_tasks)`` and a small ``T`` runs every
    task at once.  The launch is a plain
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
        slots = int(rule["ctas_per_sm"]) * sm_count(device_index)
        grid_x = min(
            slots,
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
    built on ``slots = ctas_per_sm * SM count`` under the module's taper
    (:func:`chunk_taper`); both are host-known and never depend on scheduling.
    The main kernel takes both; the dw reduce kernel a deterministic pair
    launches after it takes ``n_chunks``.
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] not in WORKSPACE_KINDS:
        raise ValueError(f"module kernel_{record['kernel']} owns no backward workspace")
    grid = record["grid_rule"]
    if grid["kind"] != "claimed_chunks":
        raise ValueError(
            f"module kernel_{record['kernel']} launches under {grid['kind']!r}, not a "
            "chunk-claiming grid"
        )
    slots = int(grid["ctas_per_sm"]) * sm_count(device_index)
    return n_chunks(rows, slots, int(grid["rows_per_issue"]), *chunk_taper(grid)), slots


def partial_capacity(record: dict[str, Any], rows: int, *, device_index: int) -> int:
    """Partial rows the workspace of one main backward module holds for every token count up to ``rows``."""

    rule = record["grid_rule"]
    if rule["kind"] != "claimed_chunks":
        raise ValueError(f"grid rule {rule['kind']!r} has no chunk table")
    return chunk_capacity(rows, int(rule["ctas_per_sm"]) * sm_count(device_index))


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
    of sums, whatever ``rows``.  The main kernel of a deterministic pair
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
