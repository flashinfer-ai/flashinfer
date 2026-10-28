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

Host plan and loader glue of the Cake BF16 RMSNorm training kernels
(SM100 / SM103 / SM107).

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

A route is the ordered list of module records one public call launches:
``fwd`` / ``fwd_residual`` launch one kernel; ``bwd`` / ``bwd_residual``
launch a pair -- the main kernel, whose persistent CTAs claim row chunks from
a counter in the workspace and write one FP32 partial row of ``dw`` per chunk,
then the dw reduce kernel under programmatic dependent launch, which folds the
partial rows into ``dw`` in chunk order.  No launch is cooperative, so a
backward never waits for the whole device.  Token counts are runtime
scalars, so a changed number of rows never recompiles anything.

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
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9": {
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
    "cake_rmsnorm_train_880768a0a264e04dbfb5": {
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
            "above": 0,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
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
    "cake_rmsnorm_train_359061bba16b8e451610": {
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
        "kernel": "cake_rmsnorm_train_359061bba16b8e451610",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_359061bba16b8e451610",
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
    "cake_rmsnorm_train_0dcdc563be377247044e": {
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
        "select_rule": None,
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__107": {
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
        "select_rule": None,
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2e8d2128a33361628509": {
        "arches": [
            "sm_100a",
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
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_60bfcb24560bbae612c3": {
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
        "select_rule": None,
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
    "cake_rmsnorm_train_d28b28659bee9c861c6c": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_d28b28659bee9c861c6c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d28b28659bee9c861c6c",
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8877322a3002e3a4fefd": {
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
    "cake_rmsnorm_train_122721bd6b4650866553": {
        "arches": [
            "sm_100a",
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
    "cake_rmsnorm_train_35a0b0ca6b2060313255": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__103": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26": {
        "arches": [
            "sm_100a",
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
            "at_most": None,
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
    "cake_rmsnorm_train_fe599c8935a3d7bc0bae__107": {
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
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fe599c8935a3d7bc0bae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fe599c8935a3d7bc0bae",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d292172df772adb17ae": {
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
            "above": 256,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_12d175079a1ea8760e0c": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_12d175079a1ea8760e0c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_12d175079a1ea8760e0c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fe599c8935a3d7bc0bae": {
        "arches": [
            "sm_100a",
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
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fe599c8935a3d7bc0bae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fe599c8935a3d7bc0bae",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_ea9bb976d8fdbfcdc387": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_ea9bb976d8fdbfcdc387",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_ea9bb976d8fdbfcdc387",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_302f64db6abe79e5259e": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_ceeef043ac64f37fc529": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__16384_inf": {
        "arches": [
            "sm_100a",
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
            "above": 16384,
            "at_most": None,
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
    "cake_rmsnorm_train_d0960be986e58f449f50__32768_inf": {
        "arches": [
            "sm_100a",
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
            "at_most": None,
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
    "cake_rmsnorm_train_c4d8be677fa70b841877__16384_inf": {
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
            "at_most": None,
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
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928": {
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
            "above": 4096,
            "at_most": 14928,
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
    "cake_rmsnorm_train_322574af704b51aef0bb": {
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
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
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
    "cake_rmsnorm_train_3e644ea19b7d14e61ab9": {
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
        "kernel": "cake_rmsnorm_train_3e644ea19b7d14e61ab9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3e644ea19b7d14e61ab9",
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
    "cake_rmsnorm_train_59270fb249dcba1ff679": {
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
        "kernel": "cake_rmsnorm_train_59270fb249dcba1ff679",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_59270fb249dcba1ff679",
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
    "cake_rmsnorm_train_5d75ea5f3c25579af496": {
        "arches": [
            "sm_100a",
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
            "sm_100a",
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
    "cake_rmsnorm_train_8e7f4d811e55a351a22e": {
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
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096": {
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
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9a830956447d4e304914": {
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
            "above": 1024,
            "at_most": 4096,
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
    "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096": {
        "arches": [
            "sm_100a",
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
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0": {
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
            "above": 256,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__256_4096": {
        "arches": [
            "sm_100a",
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
            "above": 256,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__103": {
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
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__0_4096": {
        "arches": [
            "sm_100a",
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
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_f522f5b8d4343d96ca04": {
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
            "above": 14928,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
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
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__14928_inf": {
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
            "above": 14928,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
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
    "cake_rmsnorm_train_845cdcd8145ada631461": {
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
            "at_most": None,
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
    "cake_rmsnorm_train_ee4a9053bb25b8d49fcf": {
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
            "at_most": None,
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
    "cake_rmsnorm_train_10e24516314c244771f4": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56d26996239835327501": {
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
            "at_most": None,
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
    "cake_rmsnorm_train_122721bd6b4650866553__4096_32768": {
        "arches": [
            "sm_100a",
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
    "cake_rmsnorm_train_b457c4d134e44476dc6f": {
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
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_inf": {
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
            "above": 32768,
            "at_most": None,
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
    "cake_rmsnorm_train_b93bf060d4d3defc8150": {
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
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd": {
        "arches": [
            "sm_100a",
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
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b93bf060d4d3defc8150__mid": {
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
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_16384": {
        "arches": [
            "sm_100a",
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
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc": {
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
            "above": 256,
            "at_most": None,
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
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__large": {
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
    "cake_rmsnorm_train_c73074cf44f07fcf3bb6": {
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
    "cake_rmsnorm_train_d0960be986e58f449f50": {
        "arches": [
            "sm_100a",
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
    "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_inf": {
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
            "at_most": None,
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
}
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd_h1536": [
            "cake_rmsnorm_train_7dc29629ff21625607f8",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_21c8bb55c8d672b63952",
            "cake_rmsnorm_train_b457c4d134e44476dc6f",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__32768_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
            "cake_rmsnorm_train_ceeef043ac64f37fc529",
            "cake_rmsnorm_train_880768a0a264e04dbfb5",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_d0960be986e58f449f50__16384_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_3e644ea19b7d14e61ab9",
            "cake_rmsnorm_train_59270fb249dcba1ff679",
            "cake_rmsnorm_train_9a830956447d4e304914",
            "cake_rmsnorm_train_35a0b0ca6b2060313255",
            "cake_rmsnorm_train_5d75ea5f3c25579af496",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_09601d42628de24f6f87",
            "cake_rmsnorm_train_8e7f4d811e55a351a22e",
            "cake_rmsnorm_train_359061bba16b8e451610",
            "cake_rmsnorm_train_8877322a3002e3a4fefd",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_b93bf060d4d3defc8150",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
            "cake_rmsnorm_train_b93bf060d4d3defc8150__mid",
            "cake_rmsnorm_train_fe599c8935a3d7bc0bae",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__256_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_16384",
            "cake_rmsnorm_train_ea9bb976d8fdbfcdc387",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_0dcdc563be377247044e",
            "cake_rmsnorm_train_2e8d2128a33361628509",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b44107e05796c2922187",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4__cut",
            "cake_rmsnorm_train_b44107e05796c2922187__band",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_8757f6da86abf54363d5",
            "cake_rmsnorm_train_7673978dc698790054d2",
            "cake_rmsnorm_train_3d20ab4a432bddf7349a__100",
            "cake_rmsnorm_train_3430e5d9c053309b0323__100",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace__100",
            "cake_rmsnorm_train_bad4db9153cc903b7ace__ps5",
            "cake_rmsnorm_train_288d7b1d51c340236cb7",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_308da18230017dadc02d",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
    },
    "sm_103a": {
        "bwd_h1536": [
            "cake_rmsnorm_train_7dc29629ff21625607f8",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_train_1df7db17277d363b5168",
            "cake_rmsnorm_train_f522f5b8d4343d96ca04",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724",
            "cake_rmsnorm_train_1863320f3158d35eead6",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_train_1863320f3158d35eead6__14928_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__large",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768",
            "cake_rmsnorm_train_302f64db6abe79e5259e",
            "cake_rmsnorm_train_880768a0a264e04dbfb5",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_50044ee3bd1ca54bd821",
            "cake_rmsnorm_train_ad2bb26ea5916baef963",
            "cake_rmsnorm_train_24643edc7e1143da24e0",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__103",
            "cake_rmsnorm_train_5d75ea5f3c25579af496",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26",
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
        "bwd_h6144": [
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__103",
            "cake_rmsnorm_train_322574af704b51aef0bb",
            "cake_rmsnorm_train_fe599c8935a3d7bc0bae",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__0_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_16384",
            "cake_rmsnorm_train_ea9bb976d8fdbfcdc387",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_0dcdc563be377247044e",
            "cake_rmsnorm_train_2e8d2128a33361628509",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_c5a8099e177e463daab9",
            "cake_rmsnorm_train_b44107e05796c2922187__mid",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_8757f6da86abf54363d5",
            "cake_rmsnorm_train_7673978dc698790054d2",
            "cake_rmsnorm_train_3d20ab4a432bddf7349a",
            "cake_rmsnorm_train_3430e5d9c053309b0323",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_train_288d7b1d51c340236cb7",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_308da18230017dadc02d",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
    },
    "sm_107a": {
        "bwd_h1536": [
            "cake_rmsnorm_train_7dc29629ff21625607f8__107",
            "cake_rmsnorm_train_6a428da2227972ae18ae",
            "cake_rmsnorm_train_56d26996239835327501",
            "cake_rmsnorm_train_628248febbdfe17c3325",
            "cake_rmsnorm_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_84efa2af7907877dfb56",
            "cake_rmsnorm_train_845cdcd8145ada631461__256_1024",
            "cake_rmsnorm_train_81c3d8ab0a482e3af043",
            "cake_rmsnorm_train_845cdcd8145ada631461",
            "cake_rmsnorm_train_520453292492bd2725be",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_0da5dd4b80ad9b5426ce",
            "cake_rmsnorm_train_a647fccc3d9fa9134733",
            "cake_rmsnorm_train_d41a514f5210e9adc39f",
            "cake_rmsnorm_train_ee4a9053bb25b8d49fcf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_3feed5026e1156d2fba2",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096",
            "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
            "cake_rmsnorm_train_f94861c36a14eb75e39c",
            "cake_rmsnorm_train_d28b28659bee9c861c6c",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_fe599c8935a3d7bc0bae__107",
            "cake_rmsnorm_train_5d292172df772adb17ae",
            "cake_rmsnorm_train_12d175079a1ea8760e0c",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_0dcdc563be377247044e__107",
            "cake_rmsnorm_train_60bfcb24560bbae612c3",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f__107",
            "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
            "cake_rmsnorm_train_682675bda578f5b7a312",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b44107e05796c2922187__small",
            "cake_rmsnorm_train_a4da85136fba4588239f",
            "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
            "cake_rmsnorm_train_0ad4d53c4a0df501a99d__wide",
            "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba__107",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low107",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid107",
            "cake_rmsnorm_train_bf933e1660cab8e5787b",
            "cake_rmsnorm_train_bf933e1660cab8e5787b__wide",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__107",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_7673978dc698790054d2__quad",
            "cake_rmsnorm_train_2a42e5c02b240f8709b1",
            "cake_rmsnorm_train_60f47c5d00570ecf7740",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
            "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_e646bbdbd852884953da__small",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
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


HIDDEN_SIZES: tuple[int, ...] = (512, 1536, 2048, 4096, 6144)

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


def route_key(hidden: int, kind: str, residual: bool) -> str:
    """Name the route of one public call: ``<kind>[_residual]_h<hidden>``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return f"{kind}{'_residual' if residual else ''}_h{int(hidden)}"


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


def route_applies(
    *, device_capability: Sequence[int], hidden: int, kind: str, residual: bool
) -> bool:
    """Whether the verified export owns one training call."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return route_key(hidden, kind, residual) in ROUTES.get(arch, {})


def route_modules(arch: str, hidden: int, kind: str, residual: bool) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(hidden, kind, residual)
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
    arch: str, hidden: int, kind: str, residual: bool, *, rows: int
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
        for name in route_modules(arch, hidden, kind, residual)
        if _selects(MODULES[name]["select_rule"], rows)
    )
    if not names:
        raise ValueError(
            f"no exported module of the {route_key(hidden, kind, residual)!r} route on "
            f"{arch!r} admits {rows} rows"
        )
    return names


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // int(multiple)) * int(multiple)


# The backward workspace starts with its ``uint32`` counter table (192 words at
# most, padded to four 256-byte lines); the FP32 partial rows follow.  The
# counters sit at a fixed offset of every workspace, so one buffer serves calls
# of different token counts (whose partial regions differ in size) without a
# host write between them: the main kernel resets the counter words it uses.
COUNTERS_BYTES = 1024


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
    taper, so the FP32 partial row of every chunk, and the deterministic ``dw``
    the reduce kernel folds from them in chunk order, are pure functions of
    ``(T, slots, rpi, tail_div, max_halving)`` -- never of scheduling.
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
    main launch (``slots`` its twelfth argument) and the reduce launch
    (``n_chunks``).
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
    The main kernel takes both, the dw reduce kernel launched after it takes
    ``n_chunks``.
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "chunk_partials":
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

    The workspace is one CUDA ``uint8`` buffer: the ``uint32`` counters in its
    first :data:`COUNTERS_BYTES` bytes, then the FP32 ``[capacity, hidden]``
    per-chunk dw partials, ``capacity`` the :func:`chunk_capacity` of ``rows``
    (so a buffer sized for ``rows`` serves every smaller token count; the
    counters never move).  The counters must be zero when the buffer is first
    used and the kernels leave them zero, so no host write is needed between
    launches or CUDA Graph replays.  A route whose backward modules switch on
    the row count shares one buffer: ``capacity_record`` (the module with the
    largest capacity) fixes the partial capacity for every module of the
    route, while ``n_chunks`` and ``slots`` follow ``record``, the module
    launched.
    """

    rule = record["workspace_rule"]
    chunks, slots = chunking(record, rows, device_index=device_index)
    hidden = int(record["hidden"])
    counters = int(rule["counters"])
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
    total = COUNTERS_BYTES + _round_up(partial_bytes, 256)
    return {
        "n_chunks": chunks,
        "slots": slots,
        "capacity_chunks": capacity,
        "partial_bytes": partial_bytes,
        "partial_offset": COUNTERS_BYTES,
        "counters": counters,
        "counters_offset": 0,
        "total_bytes": total,
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
    "HIDDEN_SIZES",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_HALVING",
    "MAX_ROUNDS",
    "MODULES",
    "ROUTES",
    "ROW_ALIGNMENT_ELEMENTS",
    "TAIL_DIV",
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
    "route_applies",
    "route_key",
    "route_modules",
    "run_module",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "supported_hidden_sizes",
    "workspace_layout",
]
