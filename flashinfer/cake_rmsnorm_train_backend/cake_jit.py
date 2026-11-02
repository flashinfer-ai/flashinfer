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
``dw`` in chunk order.  A window runs the pair where it is the faster form.  No
launch is cooperative, so a backward never waits for the whole device.  Token
counts are runtime scalars, so a changed number of rows never recompiles
anything.

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
    # sm_100a H1536 bf16 weight: (36864, inf] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_ae610afdf77ff6714a5b": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H1536 f32 weight: (36864, inf] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_4b6981c10fd40a9d8aec": {
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
            "above": 36864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_100a H5120 bf16 weight: (65536, inf] above T 65536, the top of the qualified range (the compensated det pair runs from T 6144 up to there)
    "cake_rmsnorm_train_2f05000ca9242ea4d6cb": {
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
        "kernel": "cake_rmsnorm_train_2f05000ca9242ea4d6cb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2f05000ca9242ea4d6cb",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_100a H5120 f32 weight: (65536, inf] above T 65536, the top of the qualified range (the compensated det pair runs from T 6144 up to there)
    "cake_rmsnorm_train_de8f6091077dc7ac5947": {
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
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 163840,
        },
    },
    # sm_100a H2048 bf16 weight: (0, 336] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_64e6aa82bb2c6d4ea978": {
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
        "kernel": "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 336,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (0, 336] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_17493054fed5409b6df9": {
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
        "kernel": "cake_rmsnorm_train_17493054fed5409b6df9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_17493054fed5409b6df9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 336,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (0, 1152] backward time 0.925-1.020x the det pair's at T 256 and 1024 over four timing sessions (B200)
    "cake_rmsnorm_train_4822793ea0c954c11af6": {
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
        "kernel": "cake_rmsnorm_train_4822793ea0c954c11af6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4822793ea0c954c11af6",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (0, 1792] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_8616070567821eaa3e35": {
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
        "kernel": "cake_rmsnorm_train_8616070567821eaa3e35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8616070567821eaa3e35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1792,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 bf16 weight: (0, 896] backward time 0.923-1.014x the det pair's at T 256 over eight timing sessions (B200)
    "cake_rmsnorm_train_080e0a6c3b18e9bad790": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (0, 1152] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_44508df7dab2ab688f85": {
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
        "kernel": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 f32 weight: (0, 1408] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_b3e6531e46667901ab91": {
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
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (0, 1408] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_864e84f3d874f850c091": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 0,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual f32 weight: (0, 1408] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_71d36617003ab7742a77": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (5632, inf] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_9433061da18b2db4299d": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 f32 weight: (3584, inf] backward time 0.970-1.001x the det pair's at T 4096, 14928, 16384, 32768 and 65536 over four timing sessions (B200)
    "cake_rmsnorm_train_a562656cef5bf5dda160": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual bf16 weight: (1408, inf] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_3fbfdd750cbeb47032b5": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
        },
    },
    # sm_100a H6144 residual f32 weight: (1408, inf] FI bench ratio < 1 against the det pair in both sessions (B200)
    "cake_rmsnorm_train_a640d0b799889f785ad5": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 196608,
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
    # sm_103a H256 bf16 weight: (4608, 5120] backward time 1.036-1.120x the det pair's at 5112 and 5120 rows in both timing sessions (GB300); above 5120 rows the deterministic package's pairs run in their own windows
    "cake_rmsnorm_train_195c4317b51bb4ea1137__4608_5120": {
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
            "above": 4608,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_103a H256 f32 weight: (4608, 5120] backward time 1.031-1.095x the det pair's at 5112 and 5120 rows in both timing sessions (GB300); above 5120 rows the deterministic package's pairs run in their own windows
    "cake_rmsnorm_train_8de0131445eed6eeabb7__4608_5120": {
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
            "above": 4608,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_103a H5120 bf16 weight: (61952, inf] above T 61952 for dw accuracy (both weight dtypes switch at T 61952: the FP32-weight route's compensated det pair, the same dw computation, reaches 0.888 of the per-element budget at T 62464 on GB300); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_9e4c0825df92af02074f": {
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
    # sm_103a H5120 f32 weight: (61952, inf] above T 61952 for dw accuracy (the compensated det pair, from T 5120, reaches 0.888 of the per-element budget at T 62464 on GB300); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_f165933caadda3434828": {
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
    # sm_103a H5120 bf16 weight: (320, 896] backward time 1.033-1.087x the det pair's at T 512 and 640 in all four timing sessions (GB300)
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__320_896": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 320,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (448, 1152] backward time 1.029-1.043x the det pair's at T 1024 in all four timing sessions (GB300)
    "cake_rmsnorm_train_44508df7dab2ab688f85__448_1152": {
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
        "kernel": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "role": "bwd_main",
        "select_rule": {
            "above": 448,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 f32 weight: (320, 1152] backward time 0.997-1.003x the det pair's at T 1024 over four timing sessions (GB300)
    "cake_rmsnorm_train_b3e6531e46667901ab91__320_1152": {
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
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 320,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (0, 2304] backward time 0.730-0.999x the det pair's at T 512 to 1536, 0.985-1.054x at T 2048, 1.016-1.063x at T 2304 and 0.971-1.142x at T 256 (GB300)
    "cake_rmsnorm_train_864e84f3d874f850c091__0_2304": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 0,
            "at_most": 2304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (320, 2304] FI bench ratio < 1 against the det pair in both sessions (GB300)
    "cake_rmsnorm_train_71d36617003ab7742a77__320_2304": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 320,
            "at_most": 2304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f64",
            "sum_bytes": 0,
        },
    },
    # sm_107a H256 bf16 weight: (3584, 36864] backward time 0.893-0.937x the det pair's at 16384 rows, 0.981-1.075x at 4096, 24576, 32768 and 32772 rows, 1.025-1.142x at 6144 and 6400 rows; 4 to 14 timing sessions per row count (R200, 212 and 216 SMs)
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
    # sm_107a H256 f32 weight: (3584, 36864] backward time 0.883-0.929x the det pair's at 16384 rows, 0.975-1.024x at 24576 rows, 1.008-1.120x at 4096, 6144, 6400, 32768 and 32772 rows; 4 to 8 timing sessions per row count (R200, 212 and 216 SMs)
    "cake_rmsnorm_train_8de0131445eed6eeabb7__3584_36864": {
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
    # sm_107a H6144 residual bf16 weight: (7168, inf] backward time 0.964-0.988x the det pair's at T 14928, 16384, 32768 and 65536 in both timing sessions on R200 with 216 SMs, 1.044-1.073x in all four with 212 SMs
    "cake_rmsnorm_train_87d2866c1ea3bcc08b13": {
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
        "kernel": "cake_rmsnorm_train_87d2866c1ea3bcc08b13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_87d2866c1ea3bcc08b13",
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
    # sm_107a H6144 residual f32 weight: (7168, inf] backward time 0.963-0.988x the det pair's at T 14928, 16384, 32768 and 65536 in both timing sessions on R200 with 216 SMs, 1.044-1.071x in all four with 212 SMs
    "cake_rmsnorm_train_647589443d4915ca8fde": {
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
        "kernel": "cake_rmsnorm_train_647589443d4915ca8fde",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_647589443d4915ca8fde",
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
    # sm_107a H6144 residual bf16 weight: (448, 3584] backward time 1.009-1.046x the det pair's at T 1024 on R200 with 212 SMs (four timing sessions) and 0.989-1.005x with 216 SMs (two)
    "cake_rmsnorm_train_864e84f3d874f850c091__448_3584": {
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
    # sm_107a H6144 residual f32 weight: (448, 3584] backward time 1.039-1.050x the det pair's at T 1024 on R200 with 212 SMs (four timing sessions) and 0.991-1.003x with 216 SMs (two)
    "cake_rmsnorm_train_71d36617003ab7742a77__448_3584": {
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
    # sm_100a H256 bf16 weight: (20480, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above this edge
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
            "above": 20480,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_100a H256 f32 weight: (20480, inf] the twosum form with an L2 warm-up: exact per-thread dw sums, so the dw error does not grow with T, and no slower than the plain form above this edge
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
            "above": 20480,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
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
    # sm_103a H1536 f32 weight: (45056, inf] above T 45056 for dw accuracy (the det pair's worst dw element reaches 0.890 of the per-element budget at T 47104 on GB300 and 0.993 at T 56832); twosum: exact per-thread dw sums
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
            "above": 45056,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H4096 bf16 weight: (50176, inf] above T 50176 for dw accuracy (the det pair's worst dw element exceeds the per-element budget at T 50688 on GB300, 1.032 of it, and reaches 0.989 of it at T 57856); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_8c8bebecc5b08454b6a0": {
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
    # sm_107a H4096 bf16 weight: (36864, inf] above T 36864 for dw accuracy (the det pair's worst dw element reaches 0.900 of the per-element budget at T 38912 on R200 with 216 SMs)
    "cake_rmsnorm_train_f11a86167db4f8e9a5da": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_107a H4096 f32 weight: (24064, inf] above T 24064 for dw accuracy (the det pair's worst dw error exceeds the FP32 eager reference's at T 24576, 0.7328 against 0.7241 of the per-element budget, and at T 29696 on R200 with 212 SMs)
    "cake_rmsnorm_train_a1f537fc94bde175bb3c": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_107a H6144 bf16 weight: (30720, inf] above T 30720 for dw accuracy (the det pair's worst dw element reaches 0.877 of the per-element budget at T 30721 and 0.978 at T 31232 on R200 with 212 SMs, and exceeds the budget at T 34816 with 216 SMs and at T 37888 with 212 SMs)
    "cake_rmsnorm_train_ddedaa3be7466a4faa97": {
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
        "kernel": "cake_rmsnorm_train_ddedaa3be7466a4faa97",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ddedaa3be7466a4faa97",
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
    # sm_107a H6144 f32 weight: (30720, inf] above T 30720 for dw accuracy (the det pair's worst dw element reaches 0.867 of the per-element budget at T 30784 on R200 with 212 SMs and exceeds the budget at T 31040 (1.057 of it) and 32768 with 212 SMs and at T 34816 with 216 SMs; its worst dw error also exceeds the FP32 eager reference's at T 30784 and 31040)
    "cake_rmsnorm_train_e77387f14390ba486055": {
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
    # sm_103a H4096 f32 weight: (23552, inf] above T 23552 for dw accuracy (over inputs 0-59 at the forward's rstd the det pair's worst error exceeds the FP32 eager reference's at T 24576 on GB300, 0.6127 against 0.6017); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_131a495c5897206b6874": {
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
            "above": 23552,
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
    # sm_107a H2048 f32 weight: (56320, inf] above T 56320 for dw accuracy (the det pair's worst dw element reaches 0.895 of the per-element budget at T 57344 in the kernel-tests harness and 0.861 at T 61952 on R200 with 212 SMs); twosum: exact per-thread dw sums
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
            "above": 56320,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_107a H1536 bf16 weight: (53248, inf] above T 53248 for dw accuracy (both weight dtypes switch at T 53248: the FP32-weight route's det pair, the same dw computation, reaches 0.895 of the per-element budget at T 53760 on R200 with 212 SMs; this route's det pair reaches 0.889 at T 54784 and 0.980 at T 62976)
    "cake_rmsnorm_train_3f75f4a81f2fb8269389": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 49152,
        },
    },
    # sm_107a H1536 f32 weight: (53248, inf] above T 53248 for dw accuracy (the det pair's worst dw element reaches 0.895 of the per-element budget at T 53760 on R200 with 212 SMs and exceeds the budget at T 61952, 1.020 of it)
    "cake_rmsnorm_train_7c0dd63f0b011049603e": {
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
    # sm_100a H2048 bf16 weight: (58880, inf] above T 58880 for dw accuracy (the det pair's worst dw element reaches 0.883 of the per-element budget at T 59392 on B200)
    "cake_rmsnorm_train_0fa4f255caf709d09f66__58880_inf": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 f32 weight: (58880, inf] above T 58880 for dw accuracy (both weight dtypes switch at T 58880: the BF16-weight route's det pair, the same dw computation, reaches 0.883 of the per-element budget at T 59392 on B200)
    "cake_rmsnorm_train_ba7dbf6a07744268705e__58880_inf": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 bf16 weight: (52736, inf] above T 52736 for dw accuracy (the det pair's worst dw element reaches 0.923 of the per-element budget at T 52737 and 0.897 at T 63488 on B200)
    "cake_rmsnorm_train_15e157d2ee9080d206a9__52736_inf": {
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
            "above": 52736,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f64",
            "sum_bytes": 131072,
        },
    },
    # sm_100a H4096 f32 weight: (52736, inf] above T 52736 for dw accuracy (both weight dtypes switch at T 52736: the BF16-weight route's det pair, the same dw computation, reaches 0.923 of the per-element budget at T 52737 on B200; this route's det pair reaches 0.891 at T 60416 and 0.983 at T 63488 in the kernel-tests harness)
    "cake_rmsnorm_train_f7573bcb748c0784f3ec__52736_inf": {
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
            "above": 52736,
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
    # sm_103a H256 bf16 weight: (114688, 139264] for dw accuracy (the det pair's worst dw element reaches 0.9475 of the per-element budget at 126976 rows and 0.8692 at 122880 rows on GB300, worst of 200 inputs of each of three families); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_172a47a15d4c1a76c061__114688_139264": {
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
        "kernel": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_172a47a15d4c1a76c061",
        "role": "bwd_main",
        "select_rule": {
            "above": 114688,
            "at_most": 139264,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_103a H256 f32 weight: (114688, 139264] for dw accuracy (one cut for both weight dtypes: the BF16-weight route's det pair reaches 0.9475 of the per-element budget at 126976 rows and 0.8692 at 122880 rows on GB300; this route's det pair reaches 0.9362 at 135168 rows, worst of 200 inputs of each of three families); twosum: exact per-thread dw sums
    "cake_rmsnorm_train_7c2aa4b429538da06a2a__114688_139264": {
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
        "kernel": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 139264,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f64",
            "sum_bytes": 4096,
        },
    },
    # sm_107a H5120 bf16 weight: the FP64-row form above T 49152 for dw accuracy (the compensated det pair, from T 19968, reaches 0.867 of the per-element budget at T 53248 on R200 with 216 SMs, above the 0.85 margin; both weight dtypes switch at T 49152)
    "cake_rmsnorm_train_f94f8c33e2c9c8e89fc3": {
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
        "kernel": "cake_rmsnorm_train_f94f8c33e2c9c8e89fc3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f94f8c33e2c9c8e89fc3",
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
    # sm_107a H5120 f32 weight: the FP64-row form above T 49152 for dw accuracy (the compensated det pair, from T 19968, reaches 0.905 of the per-element budget at T 57344 on R200 with 216 SMs, above the 0.85 margin; both weight dtypes switch at T 49152)
    "cake_rmsnorm_train_dd795e849940a49d0d29": {
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
    # The deterministic package's backward pairs (main kernel, then its dw reduce), renamed
    # into this package, for the windows the routes run them in.
    "cake_rmsnorm_train_7dc29629ff21625607f8": {
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
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
    "cake_rmsnorm_train_9b727c2c9933b2ca6724": {
        "arches": [
            "sm_100a",
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
            "above": 0,
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
    "cake_rmsnorm_train_baff9ecbafe7de1e6030": {
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
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
    "cake_rmsnorm_train_8777b1c8538527c9aaf7__32768_36864": {
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
        "kernel": "cake_rmsnorm_train_8777b1c8538527c9aaf7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8777b1c8538527c9aaf7",
        "role": "bwd_large_w32",
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
    "cake_rmsnorm_train_ceeef043ac64f37fc529__45056_58880": {
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
            "above": 45056,
            "at_most": 58880,
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
    "cake_rmsnorm_train_d0960be986e58f449f50__45056_58880": {
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
            "above": 45056,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_07f9f2fbea7163639c27__336_3072": {
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
        "kernel": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "role": "bwd_small_w32",
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
    "cake_rmsnorm_train_7e6a0b631aea3edf51f9__45056_58880": {
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
        "kernel": "cake_rmsnorm_train_7e6a0b631aea3edf51f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7e6a0b631aea3edf51f9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 45056,
            "at_most": 58880,
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
    "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_52736": {
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
            "at_most": 52736,
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
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_52736": {
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
            "at_most": 52736,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0c84999caf2b86c2e09e__1792_3584": {
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
        "kernel": "cake_rmsnorm_train_0c84999caf2b86c2e09e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0c84999caf2b86c2e09e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1792,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_626aed743358c27513f1__36864_52736": {
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
            "at_most": 52736,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__1792_3584": {
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
            "above": 1792,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_09601d42628de24f6f87": {
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
            "above": 0,
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
    "cake_rmsnorm_train_7097069b8eea6ba44d5a": {
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
            "above": 0,
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
    "cake_rmsnorm_train_9f07cea1d13b923ed91e__0_960": {
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
            "above": 0,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__0_960": {
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
            "above": 0,
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
    "cake_rmsnorm_train_c145ef195d7aeb04f242__6144_65536": {
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
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
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
    "cake_rmsnorm_train_c5831b29d17867842e5f__6144_65536": {
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
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6042554b427634669739": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_6042554b427634669739",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6042554b427634669739",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_40227b2a2b2122628d58": {
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
        "kernel": "cake_rmsnorm_train_40227b2a2b2122628d58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_40227b2a2b2122628d58",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9479b1298a1b19d4d996__6144_65536": {
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
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f": {
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
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__256_6144": {
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
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_41b80b6af97974feb181__1408_3584": {
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
            "above": 1408,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1408_3584": {
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
            "above": 1408,
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
    "cake_rmsnorm_train_109f19bb177c8f20cadc__15566_45056": {
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
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__15566_45056": {
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
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9": {
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
            "above": 0,
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
    "cake_rmsnorm_train_302f64db6abe79e5259e__32768_61952": {
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
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_880768a0a264e04dbfb5": {
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
            "above": 0,
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
    "cake_rmsnorm_train_d0960be986e58f449f50__32768_61952": {
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
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_07f9f2fbea7163639c27": {
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
            "above": 0,
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
    "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_61952": {
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
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5450d39836b26cfd3256__0_4608": {
        "arches": [
            "sm_103a",
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
            "above": 0,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_72138dce40d43da44aaf__0_4608": {
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
            "above": 0,
            "at_most": 4608,
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
    "cake_rmsnorm_train_a385ae1681a2c078344e__40960_114688": {
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
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0291949ba535c646c3a8__139264_1572864": {
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
            "above": 139264,
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
    "cake_rmsnorm_train_72138dce40d43da44aaf__12288_40960": {
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
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_72138dce40d43da44aaf__40960_114688": {
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
            "above": 40960,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0ac243f8385c32960e14__139264_1572864": {
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
            "above": 139264,
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
    "cake_rmsnorm_train_5e0c6208d4b426f79faa__0_4608": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_5e0c6208d4b426f79faa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5e0c6208d4b426f79faa",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4608,
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
    "cake_rmsnorm_train_420788e663a2c3a810f6__40960_114688": {
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
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_89505762fa1cbf33cb8f__139264_1572864": {
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
            "above": 139264,
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
    "cake_rmsnorm_train_50044ee3bd1ca54bd821": {
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
            "above": 0,
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
    "cake_rmsnorm_train_5d75ea5f3c25579af496": {
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
            "above": 0,
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
    "cake_rmsnorm_train_3fd77c8ce97e8a523abb": {
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
            "above": 0,
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
    "cake_rmsnorm_train_626aed743358c27513f1__4096_23552": {
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
            "at_most": 23552,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_23552": {
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
            "at_most": 23552,
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
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__0_320": {
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
            "above": 0,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__896_5120": {
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
            "above": 896,
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
    "cake_rmsnorm_train_c5831b29d17867842e5f__0_320": {
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
            "above": 0,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__896_5120": {
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
            "above": 896,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_79811a76205d7d7dd197": {
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
            "above": 0,
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
    "cake_rmsnorm_train_c5831b29d17867842e5f__0_5120": {
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
            "above": 0,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__0_448": {
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
            "above": 0,
            "at_most": 448,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096_103": {
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
            "above": 1152,
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
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__0_448": {
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
            "above": 0,
            "at_most": 448,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096_0_4096": {
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
            "above": 1152,
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
    "cake_rmsnorm_train_41b80b6af97974feb181__0_320": {
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
            "above": 0,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_41b80b6af97974feb181__1152_4096": {
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
            "above": 1152,
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
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__0_320": {
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
            "above": 0,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__2304_2816": {
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
            "above": 2304,
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
    "cake_rmsnorm_train_2e8d2128a33361628509__2304_2816": {
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
            "above": 2304,
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
    "cake_rmsnorm_train_56f988dde54db37f09d9__0_320": {
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
            "above": 0,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_56f988dde54db37f09d9__2304_3584": {
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
            "above": 2304,
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
    "cake_rmsnorm_train_2e8d2128a33361628509__0_320": {
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
            "above": 0,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2e8d2128a33361628509__2304_3584": {
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
            "above": 2304,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7dc29629ff21625607f8__107": {
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
            "above": 0,
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
    "cake_rmsnorm_train_628248febbdfe17c3325": {
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
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__107": {
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
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_e63beae98c236ce26335": {
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
        "kernel": "cake_rmsnorm_train_e63beae98c236ce26335",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e63beae98c236ce26335",
        "role": "bwd_mid_w32",
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
    "cake_rmsnorm_train_84efa2af7907877dfb56": {
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
            "above": 0,
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
    "cake_rmsnorm_train_520453292492bd2725be": {
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
            "above": 0,
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
    "cake_rmsnorm_train_6b13aa49e0c0414d9041": {
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
            "above": 0,
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
    "cake_rmsnorm_train_9e5ec75a15ac2bff0e4b": {
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
        "kernel": "cake_rmsnorm_train_9e5ec75a15ac2bff0e4b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e5ec75a15ac2bff0e4b",
        "role": "bwd_mid_w32",
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
    "cake_rmsnorm_train_838a9a4054731f7942af__16384_56320": {
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
            "at_most": 56320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c4d8be677fa70b841877__16384_56320": {
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
            "at_most": 56320,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5450d39836b26cfd3256__0_3584": {
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
            "above": 0,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_312bbe084eac69d858c4__0_3584": {
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
            "above": 0,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5e0c6208d4b426f79faa__0_3584": {
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
        "kernel": "cake_rmsnorm_train_5e0c6208d4b426f79faa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5e0c6208d4b426f79faa",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0da5dd4b80ad9b5426ce": {
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
            "above": 0,
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
    "cake_rmsnorm_train_69bf6e104bb48b6b74bf": {
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
            "above": 0,
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
    "cake_rmsnorm_train_0c0fa63d0ff2cdc484a4": {
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
        "kernel": "cake_rmsnorm_train_0c0fa63d0ff2cdc484a4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0c0fa63d0ff2cdc484a4",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
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
    "cake_rmsnorm_train_3feed5026e1156d2fba2": {
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
            "above": 0,
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
    "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_f94861c36a14eb75e39c": {
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
            "above": 0,
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
    "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_inf": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b841d64bbcaefff5ce7b": {
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
            "above": 0,
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
    "cake_rmsnorm_train_03efa681eb38f59b00c2": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7": {
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
            "above": 0,
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
    "cake_rmsnorm_train_2af4cf2f58ba58f78099": {
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
            "above": 0,
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
    "cake_rmsnorm_train_cd612de56c0985bb4caa": {
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
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_468cc4d1b8128187b4b3": {
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
    "cake_rmsnorm_train_5d292172df772adb17ae__0_30720": {
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
            "above": 0,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__0_30720": {
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
            "above": 0,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7d2f6707bfa3d2cd60bb__0_30720": {
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
        "kernel": "cake_rmsnorm_train_7d2f6707bfa3d2cd60bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7d2f6707bfa3d2cd60bb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__0_448": {
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
            "above": 0,
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
    "cake_rmsnorm_train_60bfcb24560bbae612c3__0_448": {
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
            "above": 0,
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
    "cake_rmsnorm_train_56f988dde54db37f09d9__0_448": {
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
            "above": 0,
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
            "cake_rmsnorm_train_172a47a15d4c1a76c061",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_8de0131445eed6eeabb7",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_09601d42628de24f6f87",
            "cake_rmsnorm_train_123db0b2f7fb38cfab03__1024_1280",
            "cake_rmsnorm_train_d0bde38aa443d42a5f04",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a",
            "cake_rmsnorm_train_fef03ce50a792c8704bd__1024_1280",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__0_960",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__0_960",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_7dc29629ff21625607f8",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_21c8bb55c8d672b63952",
            "cake_rmsnorm_train_b457c4d134e44476dc6f__32768_36864",
            "cake_rmsnorm_train_ae610afdf77ff6714a5b",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_36864",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_baff9ecbafe7de1e6030",
            "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
            "cake_rmsnorm_train_4bf450b6af0314ed03be",
            "cake_rmsnorm_train_8777b1c8538527c9aaf7__32768_36864",
            "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_36864",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__336_3072",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__4096_7168",
            "cake_rmsnorm_train_0fa4f255caf709d09f66",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_13608",
            "cake_rmsnorm_train_0fa4f255caf709d09f66__13608_45056",
            "cake_rmsnorm_train_ceeef043ac64f37fc529__45056_58880",
            "cake_rmsnorm_train_0fa4f255caf709d09f66__58880_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_7168",
            "cake_rmsnorm_train_d0960be986e58f449f50__11264_13608",
            "cake_rmsnorm_train_d0960be986e58f449f50__45056_58880",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_17493054fed5409b6df9",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__336_3072",
            "cake_rmsnorm_train_d96a14fe8882bbd9ec2e",
            "cake_rmsnorm_train_b187e5989302def22b2d__4096_15403",
            "cake_rmsnorm_train_ba7dbf6a07744268705e",
            "cake_rmsnorm_train_7e6a0b631aea3edf51f9__45056_58880",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__58880_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_15403",
            "cake_rmsnorm_train_d0960be986e58f449f50__45056_58880",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_4822793ea0c954c11af6",
            "cake_rmsnorm_train_9a830956447d4e304914__1152_4096",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_4608",
            "cake_rmsnorm_train_15e157d2ee9080d206a9",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_52736",
            "cake_rmsnorm_train_15e157d2ee9080d206a9__52736_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1152_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_4608",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_52736",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_8616070567821eaa3e35",
            "cake_rmsnorm_train_0c84999caf2b86c2e09e__1792_3584",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec",
            "cake_rmsnorm_train_626aed743358c27513f1__36864_52736",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec__52736_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1792_3584",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_52736",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_080e0a6c3b18e9bad790",
            "cake_rmsnorm_train_b482c84dbffcd75a4a4c__896_6144",
            "cake_rmsnorm_train_c145ef195d7aeb04f242__6144_65536",
            "cake_rmsnorm_train_2f05000ca9242ea4d6cb",
            "cake_rmsnorm_train_c5831b29d17867842e5f__896_6144",
            "cake_rmsnorm_train_c5831b29d17867842e5f__6144_65536",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_6042554b427634669739",
            "cake_rmsnorm_train_40227b2a2b2122628d58",
            "cake_rmsnorm_train_9479b1298a1b19d4d996__6144_65536",
            "cake_rmsnorm_train_de8f6091077dc7ac5947",
            "cake_rmsnorm_train_c5831b29d17867842e5f",
            "cake_rmsnorm_train_c5831b29d17867842e5f__256_6144",
            "cake_rmsnorm_train_c5831b29d17867842e5f__6144_65536",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_44508df7dab2ab688f85",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096",
            "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_5632",
            "cake_rmsnorm_train_9433061da18b2db4299d",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_5632",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_864e84f3d874f850c091",
            "cake_rmsnorm_train_3fbfdd750cbeb47032b5",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_b3e6531e46667901ab91",
            "cake_rmsnorm_train_41b80b6af97974feb181__1408_3584",
            "cake_rmsnorm_train_a562656cef5bf5dda160",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1408_3584",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_71d36617003ab7742a77",
            "cake_rmsnorm_train_a640d0b799889f785ad5",
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
            "cake_rmsnorm_train_5450d39836b26cfd3256__0_4608",
            "cake_rmsnorm_train_195c4317b51bb4ea1137__4608_5120",
            "cake_rmsnorm_train_a385ae1681a2c078344e__5120_12288",
            "cake_rmsnorm_train_af005462cd6a301f31c4__12288_40960",
            "cake_rmsnorm_train_a385ae1681a2c078344e__40960_114688",
            "cake_rmsnorm_train_172a47a15d4c1a76c061__114688_139264",
            "cake_rmsnorm_train_0291949ba535c646c3a8__139264_1572864",
            "cake_rmsnorm_train_c18297d75c0d424978fd",
            "cake_rmsnorm_train_72138dce40d43da44aaf__0_4608",
            "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288",
            "cake_rmsnorm_train_72138dce40d43da44aaf__12288_40960",
            "cake_rmsnorm_train_72138dce40d43da44aaf__40960_114688",
            "cake_rmsnorm_train_0ac243f8385c32960e14__139264_1572864",
            "cake_rmsnorm_train_e73321cec89d33336ea1",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_5e0c6208d4b426f79faa__0_4608",
            "cake_rmsnorm_train_8de0131445eed6eeabb7__4608_5120",
            "cake_rmsnorm_train_420788e663a2c3a810f6__5120_12288",
            "cake_rmsnorm_train_b7d9577ff884de98ea06__12288_40960",
            "cake_rmsnorm_train_420788e663a2c3a810f6__40960_114688",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__114688_139264",
            "cake_rmsnorm_train_89505762fa1cbf33cb8f__139264_1572864",
            "cake_rmsnorm_train_f788ef57163552825d27",
            "cake_rmsnorm_train_72138dce40d43da44aaf__0_4608",
            "cake_rmsnorm_train_72138dce40d43da44aaf__5120_12288",
            "cake_rmsnorm_train_72138dce40d43da44aaf__12288_40960",
            "cake_rmsnorm_train_72138dce40d43da44aaf__40960_114688",
            "cake_rmsnorm_train_0ac243f8385c32960e14__139264_1572864",
            "cake_rmsnorm_train_e73321cec89d33336ea1",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_09601d42628de24f6f87",
            "cake_rmsnorm_train_8e7f4d811e55a351a22e",
            "cake_rmsnorm_train_274b0055aace183f2367",
            "cake_rmsnorm_train_cfaae3cec98f8501832b",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_train_122721bd6b4650866553__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__0_960",
            "cake_rmsnorm_train_56ee8e66479625fd078a__960_32768",
            "cake_rmsnorm_train_a546f410ec8cf03ef20d",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__0_960",
            "cake_rmsnorm_train_122721bd6b4650866553__960_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_7dc29629ff21625607f8",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_1df7db17277d363b5168",
            "cake_rmsnorm_train_4df08ce51fb01ba1193f",
            "cake_rmsnorm_train_f522f5b8d4343d96ca04__15566_46592",
            "cake_rmsnorm_train_327bfeb4ab8fefc2731d",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_train_1863320f3158d35eead6__15566_46592",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_baff9ecbafe7de1e6030",
            "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
            "cake_rmsnorm_train_a99e2b380514c4d86d60",
            "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2",
            "cake_rmsnorm_train_109f19bb177c8f20cadc__15566_45056",
            "cake_rmsnorm_train_963a7ca8d03f78bbe6f9",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_train_1863320f3158d35eead6__15566_45056",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__large",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768",
            "cake_rmsnorm_train_302f64db6abe79e5259e__32768_61952",
            "cake_rmsnorm_train_8bd89a4c834f80db8a34",
            "cake_rmsnorm_train_880768a0a264e04dbfb5",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_61952",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_07f9f2fbea7163639c27",
            "cake_rmsnorm_train_d96a14fe8882bbd9ec2e",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__large",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__16384_32768",
            "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_61952",
            "cake_rmsnorm_train_88c6c6b2af1922f4c1a9",
            "cake_rmsnorm_train_880768a0a264e04dbfb5",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_61952",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_50044ee3bd1ca54bd821",
            "cake_rmsnorm_train_ad2bb26ea5916baef963",
            "cake_rmsnorm_train_24643edc7e1143da24e0",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_50176",
            "cake_rmsnorm_train_8c8bebecc5b08454b6a0",
            "cake_rmsnorm_train_5d75ea5f3c25579af496",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_50176",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_3fd77c8ce97e8a523abb",
            "cake_rmsnorm_train_16545056022785f2a7d8",
            "cake_rmsnorm_train_78c26a53e83c9803bcbf",
            "cake_rmsnorm_train_626aed743358c27513f1__4096_23552",
            "cake_rmsnorm_train_131a495c5897206b6874",
            "cake_rmsnorm_train_5d75ea5f3c25579af496",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_23552",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__0_320",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__320_896",
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__896_5120",
            "cake_rmsnorm_train_7ffb443a63a92ab6b0e0__5120_61952",
            "cake_rmsnorm_train_9e4c0825df92af02074f",
            "cake_rmsnorm_train_c5831b29d17867842e5f__0_320",
            "cake_rmsnorm_train_c5831b29d17867842e5f__896_5120",
            "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_79811a76205d7d7dd197",
            "cake_rmsnorm_train_f9f7d4cd75ef91cc8374__5120_61952",
            "cake_rmsnorm_train_f165933caadda3434828",
            "cake_rmsnorm_train_c5831b29d17867842e5f__0_5120",
            "cake_rmsnorm_train_c5831b29d17867842e5f__5120_61952",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__0_448",
            "cake_rmsnorm_train_44508df7dab2ab688f85__448_1152",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096_103",
            "cake_rmsnorm_train_322574af704b51aef0bb__4096_49152",
            "cake_rmsnorm_train_582b2a20f9be648af8ec",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__0_448",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096_0_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_864e84f3d874f850c091__0_2304",
            "cake_rmsnorm_train_0dcdc563be377247044e__2304_2816",
            "cake_rmsnorm_train_115dcc719a6de4b079f1",
            "cake_rmsnorm_train_0dcdc563be377247044e__4608_5632",
            "cake_rmsnorm_train_115dcc719a6de4b079f1__5632_32768",
            "cake_rmsnorm_train_52831494c1b62db758fd",
            "cake_rmsnorm_train_2e8d2128a33361628509__2304_2816",
            "cake_rmsnorm_train_2e8d2128a33361628509__4608_5632",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_41b80b6af97974feb181__0_320",
            "cake_rmsnorm_train_b3e6531e46667901ab91__320_1152",
            "cake_rmsnorm_train_41b80b6af97974feb181__1152_4096",
            "cake_rmsnorm_train_9a8c0396edd85d4d545e__4096_49152",
            "cake_rmsnorm_train_4f9a76e68671f213ab7c",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__0_320",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096_0_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_56f988dde54db37f09d9__0_320",
            "cake_rmsnorm_train_71d36617003ab7742a77__320_2304",
            "cake_rmsnorm_train_56f988dde54db37f09d9__2304_3584",
            "cake_rmsnorm_train_3c390339fa8c82311a0a",
            "cake_rmsnorm_train_56f988dde54db37f09d9__4608_5632",
            "cake_rmsnorm_train_3c390339fa8c82311a0a__5632_15872",
            "cake_rmsnorm_train_56c46a201c168050aa60",
            "cake_rmsnorm_train_2e8d2128a33361628509__0_320",
            "cake_rmsnorm_train_2e8d2128a33361628509__2304_3584",
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
            "cake_rmsnorm_train_5450d39836b26cfd3256__0_3584",
            "cake_rmsnorm_train_195c4317b51bb4ea1137__3584_36864",
            "cake_rmsnorm_train_172a47a15d4c1a76c061__36864_inf",
            "cake_rmsnorm_train_312bbe084eac69d858c4__0_3584",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_5e0c6208d4b426f79faa__0_3584",
            "cake_rmsnorm_train_8de0131445eed6eeabb7__3584_36864",
            "cake_rmsnorm_train_7c2aa4b429538da06a2a__36864_inf",
            "cake_rmsnorm_train_312bbe084eac69d858c4__0_3584",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_3feed5026e1156d2fba2",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096",
            "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
            "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3",
            "cake_rmsnorm_train_f94861c36a14eb75e39c",
            "cake_rmsnorm_train_d75459d901441384398b",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_inf",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_b841d64bbcaefff5ce7b",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22__1024_4096",
            "cake_rmsnorm_train_383f709cd5c4fadd9436",
            "cake_rmsnorm_train_03efa681eb38f59b00c2",
            "cake_rmsnorm_train_f94861c36a14eb75e39c",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_inf",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_7dc29629ff21625607f8__107",
            "cake_rmsnorm_train_6a428da2227972ae18ae",
            "cake_rmsnorm_train_56d26996239835327501__16384_53248",
            "cake_rmsnorm_train_3f75f4a81f2fb8269389",
            "cake_rmsnorm_train_628248febbdfe17c3325",
            "cake_rmsnorm_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_53248",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__107",
            "cake_rmsnorm_train_e63beae98c236ce26335",
            "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_53248",
            "cake_rmsnorm_train_7c0dd63f0b011049603e",
            "cake_rmsnorm_train_628248febbdfe17c3325",
            "cake_rmsnorm_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_53248",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_84efa2af7907877dfb56",
            "cake_rmsnorm_train_845cdcd8145ada631461__256_1024",
            "cake_rmsnorm_train_81c3d8ab0a482e3af043",
            "cake_rmsnorm_train_845cdcd8145ada631461__16384_50176",
            "cake_rmsnorm_train_1a3df5ac020055477581",
            "cake_rmsnorm_train_520453292492bd2725be",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50176",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_6b13aa49e0c0414d9041",
            "cake_rmsnorm_train_838a9a4054731f7942af__256_1024",
            "cake_rmsnorm_train_9e5ec75a15ac2bff0e4b",
            "cake_rmsnorm_train_838a9a4054731f7942af__16384_56320",
            "cake_rmsnorm_train_810989bdea464df479ce",
            "cake_rmsnorm_train_520453292492bd2725be",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_56320",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_0da5dd4b80ad9b5426ce",
            "cake_rmsnorm_train_a647fccc3d9fa9134733",
            "cake_rmsnorm_train_d41a514f5210e9adc39f",
            "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_36864",
            "cake_rmsnorm_train_f11a86167db4f8e9a5da",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_36864",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_0c0fa63d0ff2cdc484a4",
            "cake_rmsnorm_train_dc439f8335327e482025",
            "cake_rmsnorm_train_b08633208e60b8ce94fd",
            "cake_rmsnorm_train_0f1859b9fe9e47668a1c__4096_24064",
            "cake_rmsnorm_train_a1f537fc94bde175bb3c",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_24064",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_838c077df3a6d35d55b7",
            "cake_rmsnorm_train_6816fb80e97380f1bcbc",
            "cake_rmsnorm_train_b8b0913b40f035f43b76__19968_49152",
            "cake_rmsnorm_train_f94f8c33e2c9c8e89fc3",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_cd612de56c0985bb4caa",
            "cake_rmsnorm_train_468cc4d1b8128187b4b3",
            "cake_rmsnorm_train_7b9e10b21b75d4615135__19968_49152",
            "cake_rmsnorm_train_dd795e849940a49d0d29",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_5d292172df772adb17ae__0_30720",
            "cake_rmsnorm_train_ddedaa3be7466a4faa97",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__0_30720",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_0dcdc563be377247044e__0_448",
            "cake_rmsnorm_train_864e84f3d874f850c091__448_3584",
            "cake_rmsnorm_train_0dcdc563be377247044e__3584_7168",
            "cake_rmsnorm_train_87d2866c1ea3bcc08b13",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__0_448",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_7d2f6707bfa3d2cd60bb__0_30720",
            "cake_rmsnorm_train_e77387f14390ba486055",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__0_30720",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_56f988dde54db37f09d9__0_448",
            "cake_rmsnorm_train_71d36617003ab7742a77__448_3584",
            "cake_rmsnorm_train_56f988dde54db37f09d9__3584_7168",
            "cake_rmsnorm_train_647589443d4915ca8fde",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__0_448",
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
    grid is ``min(slots, n_chunks)``; the launch is a plain
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
            n_chunks(rows, slots, int(rule["rows_per_issue"]), *chunk_taper(rule)),
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
