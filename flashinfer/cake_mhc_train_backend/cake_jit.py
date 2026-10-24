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

Host plan and loader glue of the Cake four-stream mHC training kernels
(hidden 4096, 4 streams; SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch,
grid, selection and workspace rules and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is the ordered list of module records one public call launches:
``fwd_prepare`` (fn split, mapping GEMM, coefficients, collapse), ``fwd_recompute``
(the strict prefix of ``fwd_prepare`` that regenerates the per-token statistics
bitwise for checkpoint replay), ``fwd_finish`` (residual mixing), ``bwd_prepare``
(``dB`` and the per-token gradient statistics) and ``bwd_finish`` (transposed fn
split, token-owner pass, column-slab pass, fixed-order final reduction).  The
token count ``T`` is a runtime scalar: a changed number of rows never
recompiles anything, and ``T == 0`` launches nothing.

A module record is one (kernel, launch policy) pair.  A kernel launched under
several policies has one record per policy (``<kernel>__<stage>`` after the
first), all naming the kernel's ``library``, so it is compiled once: the fn
split runs once per forward route (its forward operand layout) and once per
backward route (``__t``, the transposed layout, selected by the ``want_t``
scalar of :data:`RECORD_SCALARS`).

The grid, chunk and workspace rules are pure integer functions of ``(T, SM
count)`` (:func:`k1_units`, :func:`k7_chunks`, :func:`k6_grid`, :func:`k2_grid`,
:func:`k3_grid`, :func:`k4_grid`, :func:`k5_grid`; :func:`row_grid` is the
persistent row grid of the design table, kept for the rules table), shared
verbatim with the kernels' own test references, so the work partition -- and
with it every fixed-order reduction -- is a pure function of ``(T, device)``.

Programmatic dependent launch is a launch attribute of the generated binding,
recorded per module as ``launch.use_pdl``: it is set exactly for the kernels
whose device code waits for the preceding grid (``griddepcontrol.wait``) before
reading a predecessor's output -- the coefficient pass, the row streams, the
token-owner pass, the column-slab pass and the final reduction -- and left off
for the fn split and the mapping pass, which carry no such wait.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Optional, Sequence

# Filled mechanically from the complete verified program bundle (the derived
# leaves ``kernel_symbol``, ``arg_plan``, ``logical`` and the launch leaves
# ``block`` / ``cluster`` / ``cooperative`` / ``dynamic_smem_bytes`` /
# ``use_pdl``); everything else is the host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_mhc_train_fn_split": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnp"],
            ["buffer", "fnTp"],
            ["parameter", "want_t"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fn_split",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fn_split",
        "logical": {
            "fn": "fn",
            "fnTp": "fnTp",
            "fnp": "fnp",
            "want_t": "want_t",
        },
        "role": "fn_split_fwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fn_split__t": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "fn"],
            ["buffer", "fnp"],
            ["buffer", "fnTp"],
            ["parameter", "want_t"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fn_split",
        "kernel_symbol": "kernel_cake_mhc_train_fn_split",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fn_split",
        "logical": {
            "fn": "fn",
            "fnTp": "fnTp",
            "fnp": "fnp",
            "want_t": "want_t",
        },
        "role": "fn_split_bwd",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_fwd_map": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "FNP"],
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["parameter", "T"],
            ["parameter", "m_tiles"],
            ["parameter", "log2_pk"],
            ["buffer", "ticket"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 2,
            "kind": "units_bounded",
            "tile_rows": 128,
            "units_rule": "k1_units",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_map",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_map",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 115712,
            "persistent_ctas_per_sm": 2,
            "use_pdl": False,
        },
        "library": "cake_mhc_train_fwd_map",
        "logical": {
            "FNP": "fnp",
            "T": "rows",
            "X": "x_flat",
            "log2_pk": "log2_pk",
            "m_tiles": "m_tiles",
            "qpart": "qpart",
            "spart": "spart",
            "ticket": "ticket",
        },
        "role": "fwd_map",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_coef": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "spart"],
            ["buffer", "qpart"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "stats"],
            ["parameter", "T"],
            ["parameter", "p_k"],
            ["parameter", "t_pad"],
            ["parameter", "rms_eps"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tiles",
            "rows_per_tile": 16,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_coef",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_coef",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_coef",
        "logical": {
            "T": "rows",
            "base": "base",
            "hc_eps": "hc_eps",
            "p_k": "p_k",
            "qpart": "qpart",
            "rms_eps": "rms_eps",
            "scale": "scale",
            "spart": "spart",
            "stats": "stats",
            "t_pad": "t_pad",
        },
        "role": "fwd_coef",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "fwd",
        },
    },
    "cake_mhc_train_fwd_collapse": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "u"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 2,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_collapse",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_collapse",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_collapse",
        "logical": {
            "T": "rows",
            "stats": "stats",
            "u": "u",
            "x": "x",
        },
        "role": "fwd_collapse",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_fwd_finish": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "b"],
            ["buffer", "stats"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 2,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_fwd_finish",
        "kernel_symbol": "kernel_cake_mhc_train_fwd_finish",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_fwd_finish",
        "logical": {
            "T": "rows",
            "b": "b",
            "stats": "stats",
            "x": "x",
            "y": "y",
        },
        "role": "fwd_finish",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_prepare": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "b"],
            ["buffer", "x"],
            ["buffer", "stats"],
            ["buffer", "db"],
            ["buffer", "gstats"],
            ["parameter", "T"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "col_split": 1,
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_prepare",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_prepare",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1280,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_prepare",
        "logical": {
            "T": "rows",
            "b": "b",
            "db": "db",
            "dy": "dy",
            "gstats": "gstats",
            "stats": "stats",
            "x": "x",
        },
        "role": "bwd_prepare",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_mhc_train_bwd_token": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dU"],
            ["buffer", "stats"],
            ["buffer", "gstats"],
            ["buffer", "scale"],
            ["buffer", "base"],
            ["buffer", "dzp"],
            ["buffer", "tokp"],
            ["buffer", "param_partial"],
            ["parameter", "T"],
            ["parameter", "t_pad"],
            ["parameter", "rows_per_cta"],
            ["parameter", "hc_eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k6_grid",
            "ctas_per_sm": 2,
            "group_rows": 8,
            "kind": "chunks",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_token",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_token",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 91136,
            "persistent_ctas_per_sm": 2,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_token",
        "logical": {
            "T": "rows",
            "X": "x",
            "base": "base",
            "dU": "du",
            "dzp": "dzp",
            "gstats": "gstats",
            "hc_eps": "hc_eps",
            "param_partial": "param_partial",
            "rows_per_cta": "rows_per_cta",
            "scale": "scale",
            "stats": "stats",
            "t_pad": "t_pad",
            "tokp": "tokp",
        },
        "role": "bwd_token",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_slab": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "X"],
            ["buffer", "dY"],
            ["buffer", "dU"],
            ["buffer", "dzp"],
            ["buffer", "fnTp"],
            ["buffer", "dX"],
            ["buffer", "tokp"],
            ["buffer", "dfn_part"],
            ["parameter", "T"],
            ["parameter", "P"],
            ["parameter", "rpc"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunks_rule": "k7_chunks",
            "kind": "slab_chunks",
            "slabs": 64,
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_slab",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_slab",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 222208,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_slab",
        "logical": {
            "P": "parts",
            "T": "rows",
            "X": "x_flat",
            "dU": "du",
            "dX": "dx_flat",
            "dY": "dy_flat",
            "dfn_part": "dfn_part",
            "dzp": "dzp_rows",
            "fnTp": "fnTp",
            "rpc": "rows_per_chunk",
            "tokp": "tokp",
        },
        "role": "bwd_slab",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
    "cake_mhc_train_bwd_reduce": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dfn_part"],
            ["buffer", "param_partial"],
            ["buffer", "dfn"],
            ["buffer", "dscale"],
            ["buffer", "dbase"],
            ["parameter", "n_parts"],
            ["parameter", "n_chunks"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 385,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_mhc_train_bwd_reduce",
        "kernel_symbol": "kernel_cake_mhc_train_bwd_reduce",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 65536,
            "persistent_ctas_per_sm": 1,
            "use_pdl": True,
        },
        "library": "cake_mhc_train_bwd_reduce",
        "logical": {
            "accumulate": "accumulate",
            "dbase": "dbase",
            "dfn": "dfn",
            "dfn_part": "dfn_part",
            "dscale": "dscale",
            "n_chunks": "chunks",
            "n_parts": "parts",
            "param_partial": "param_partial",
        },
        "role": "bwd_reduce",
        "select_rule": None,
        "workspace_rule": {
            "kind": "regions",
            "workspace": "bwd",
        },
    },
}

# Route key -> ordered module records, per architecture.  Every architecture
# ships the same nine programs under the same plan (the device code of each
# program is folded over the architectures by the exporter).
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd_finish_h4096": [
            "cake_mhc_train_fn_split__t",
            "cake_mhc_train_bwd_token",
            "cake_mhc_train_bwd_slab",
            "cake_mhc_train_bwd_reduce",
        ],
        "bwd_prepare_h4096": [
            "cake_mhc_train_bwd_prepare",
        ],
        "fwd_finish_h4096": [
            "cake_mhc_train_fwd_finish",
        ],
        "fwd_prepare_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map",
            "cake_mhc_train_fwd_coef",
            "cake_mhc_train_fwd_collapse",
        ],
        "fwd_recompute_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map",
            "cake_mhc_train_fwd_coef",
        ],
    },
    "sm_103a": {
        "bwd_finish_h4096": [
            "cake_mhc_train_fn_split__t",
            "cake_mhc_train_bwd_token",
            "cake_mhc_train_bwd_slab",
            "cake_mhc_train_bwd_reduce",
        ],
        "bwd_prepare_h4096": [
            "cake_mhc_train_bwd_prepare",
        ],
        "fwd_finish_h4096": [
            "cake_mhc_train_fwd_finish",
        ],
        "fwd_prepare_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map",
            "cake_mhc_train_fwd_coef",
            "cake_mhc_train_fwd_collapse",
        ],
        "fwd_recompute_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map",
            "cake_mhc_train_fwd_coef",
        ],
    },
    "sm_107a": {
        "bwd_finish_h4096": [
            "cake_mhc_train_fn_split__t",
            "cake_mhc_train_bwd_token",
            "cake_mhc_train_bwd_slab",
            "cake_mhc_train_bwd_reduce",
        ],
        "bwd_prepare_h4096": [
            "cake_mhc_train_bwd_prepare",
        ],
        "fwd_finish_h4096": [
            "cake_mhc_train_fwd_finish",
        ],
        "fwd_prepare_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map",
            "cake_mhc_train_fwd_coef",
            "cake_mhc_train_fwd_collapse",
        ],
        "fwd_recompute_h4096": [
            "cake_mhc_train_fn_split",
            "cake_mhc_train_fwd_map",
            "cake_mhc_train_fwd_coef",
        ],
    },
}
HIDDEN_SIZES: tuple[int, ...] = (4096,)

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
KINDS = ("fwd_prepare", "fwd_recompute", "fwd_finish", "bwd_prepare", "bwd_finish")
# The routes that hold a caller-owned workspace, by the ``workspace`` name of
# their records' ``workspace_rule``.
WORKSPACE_OF_KIND: dict[str, str] = {
    "fwd_prepare": "fwd",
    "fwd_recompute": "fwd",
    "bwd_finish": "bwd",
}

# Model constants the kernels are built for (``_static_`` dimensions).
STREAMS = 4
HIDDEN = 4096
HIDDEN_TOTAL = 16384  # the flattened token: 4 streams x 4096
MAPPING_ROWS = 24  # 4 pre logits + 4 post logits + 16 Sinkhorn logits
N_PAD = 32  # mapping rows padded to the tensor-core N of the mapping GEMMs
PIECES = 3  # BF16 pieces of an FP32 operand (exact three-way split)
SLAB_D = 64  # columns per stream of one column-slab unit of the backward
TILE_T = 128  # token tile of the forward mapping pass (``T_pad`` granularity)
SLAB_TILE_T = 64  # token tile of the column-slab pass
RMS_EPS = 1e-5
HC_EPS = 1e-6
SINKHORN_ITERS = 20

# Per-token records (FP32 columns).  ``stats`` is retained between the forward
# and the backward (256 bytes per token); ``gstats`` is written by
# ``bwd_prepare`` (128 bytes per token).
STATS_COLS = 64
GSTATS_COLS = 32
TOKP_COLS = 24
DZP_COLS = 96  # three BF16 pieces of dz' = dz * r, 32 columns each
PARAM_COLS = 32
TICKET_WORDS = 64  # one 256-byte region of uint32, zero on first use; only the opt-in ticket schedule of the mapping pass reads word 0 (the shipped static form leaves it unread)
STATS_LAYOUT: dict[str, tuple[int, int]] = {
    "r": (0, 1),
    "z": (1, 25),
    "pre": (25, 29),
    "post": (29, 33),
    "C": (33, 49),
    "pad": (49, 64),
}
GSTATS_LAYOUT: dict[str, tuple[int, int]] = {
    "dpost": (0, 4),
    "dC": (4, 20),
    "pad": (20, 32),
}
TOKP_LAYOUT: dict[str, tuple[int, int]] = {
    "r": (0, 1),
    "c_t": (1, 2),
    "pre": (2, 6),
    "C": (6, 22),
    "pad": (22, 24),
}
PARAM_LAYOUT: dict[str, tuple[int, int]] = {
    "da_sum": (0, 24),
    "dscale_terms": (24, 27),
    "pad": (27, 32),
}

# The caller-owned workspaces (``workspace_rule`` kind ``regions``): ordered
# regions, each rounded up to 256 bytes; a shape entry that is a string names
# a host-plan quantity of ``host_plan`` (``t_pad``, ``p_k``, ``parts``, ``chunks``).
WORKSPACE_REGIONS: dict[str, list[dict[str, Any]]] = {
    "fwd": [
        {"dtype": "bfloat16", "name": "fnp", "shape": [3, 32, 16384]},
        {"dtype": "float32", "name": "spart", "shape": ["p_k", "t_pad", 24]},
        {"dtype": "float32", "name": "qpart", "shape": ["p_k", "t_pad"]},
        {"dtype": "uint32", "name": "ticket", "shape": [64]},
    ],
    "bwd": [
        {"dtype": "bfloat16", "name": "fnTp", "shape": [3, 16384, 32]},
        {"dtype": "bfloat16", "name": "dzp", "shape": ["t_pad", 96]},
        {"dtype": "float32", "name": "tokp", "shape": ["t_pad", 24]},
        {"dtype": "float32", "name": "param_partial", "shape": ["chunks", 32]},
        {"dtype": "float32", "name": "dfn_part", "shape": ["parts", 24, 16384]},
        {"dtype": "uint32", "name": "ticket", "shape": [64]},
    ],
}
# Scalar values a record fixes itself (the launch policy of a stage record):
# the fn split writes the forward operand layout (``want_t = 0``) or the
# transposed layout of the backward (``want_t = 1``).
RECORD_SCALARS: dict[str, dict[str, int]] = {
    "cake_mhc_train_fn_split": {"want_t": 0},
    "cake_mhc_train_fn_split__t": {"want_t": 1},
}
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).  The
# ``*_flat`` names are the ``[T, 16384]`` views of the ``[T, 4, 4096]`` tensors
# (the tensor-core kernels describe them as two-dimensional); ``dzp_rows`` is
# the first ``T`` rows of the padded ``dzp`` region.
LOGICAL_TENSORS = (
    "x",
    "x_flat",
    "fn",
    "fnp",
    "fnTp",
    "scale",
    "base",
    "u",
    "stats",
    "b",
    "y",
    "dy",
    "dy_flat",
    "db",
    "gstats",
    "du",
    "dx_flat",
    "dzp",
    "dzp_rows",
    "tokp",
    "param_partial",
    "dfn_part",
    "spart",
    "qpart",
    "ticket",
    "dfn",
    "dscale",
    "dbase",
)
LOGICAL_SCALARS = (
    "rows",
    "t_pad",
    "m_tiles",
    "log2_pk",
    "p_k",
    "parts",
    "rows_per_chunk",
    "chunks",
    "rows_per_cta",
    "rms_eps",
    "hc_eps",
    "want_t",
    "accumulate",
)
# The host-rule table the kernels' test references and this registry are
# checked against (the same rows, in the same column order).
RULES_SM_COUNTS = (148, 152, 212)
RULES_T_VALUES = (
    0, 1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 255, 1000, 1024, 2049, 4096, 8192,
    14928, 15288, 15566, 15590, 16114, 16132, 16164, 16263, 20001, 32768, 65536,
)  # fmt: skip


# ----------------------------------------------------------------------------
# Host rules (integer arithmetic only; valid for T >= 1).  These are the text
# of record shared with the kernels' test references: a change here is a
# change of the work partition and of every fixed-order reduction.


def k1_units(T, sm_count):  # -> (P_K, units, grid)
    mb = -(-T // 128)
    best = None
    for pk in (8, 16, 32):
        units = mb * pk
        rounds = -(-units // sm_count)
        grid = min(
            2 * sm_count, units
        )  # SM-level rounds: both CTAs of an SM share its bandwidth and the ticket queue
        cost = (
            rounds * sm_count * 128 * (16384 // pk + 100) + 100 * pk * T
        )  # bytes-equivalent makespan + K2 partial reads
        if best is None or (cost, pk) < best[0]:
            best = ((cost, pk), pk, units, grid)
    return best[1:]


def k7_chunks(T, sm_count, slab_d=64):  # -> (P, units, grid, rows_per_chunk)
    S = 16384 // (4 * slab_d)
    tiles = -(-T // 64)
    b_unit = 3 * 4 * slab_d * 2 + slab_d * 2 + 192 + 96
    best = None
    for p in range(1, min(32, tiles) + 1):
        rpc = 64 * (-(-tiles // p))
        units = S * p
        grid = min(sm_count, units)
        cost = (
            sm_count * (-(-units // sm_count)) * rpc * b_unit + 2 * p * 1_572_864
        )  # makespan bytes + partial write+read
        if best is None or (cost, p) < best[0]:
            best = ((cost, p), p, units, grid, rpc)
    return best[1:]


def k6_grid(T, sm_count):  # static token chunks for the parameter partials -> (G6, R)
    groups = -(-T // 8)
    G6 = min(2 * sm_count, groups)
    R = 8 * (-(-groups // G6))
    return G6, R


def row_grid(
    T, sm_count, grid_per_sm
):  # K3, K4, K5: the design's persistent grid (M1 form; a GRID_PER_SM > 0 entry)
    return min(T, grid_per_sm * sm_count)


def k2_grid(T):
    return -(-T // 16)


# The M2 production forms of the row streams launch one trip per CTA (PD17; the modules' ``grid_rule`` with
# ``GRID_PER_SM = 0``): ``ceil(T col_split / rows_per_cta)`` CTAs of 256 threads, the grid of ``module.grid_for``.
def row_cta_grid(
    T, rows_per_cta=1, col_split=1
):  # one (sub-)token per thread group and trip
    return -(-(T * col_split) // rows_per_cta)


def k3_grid(T):  # fwd_collapse production (rows_per_cta=1, col_split=2): 2 T CTAs
    return row_cta_grid(T, 1, 2)


def k4_grid(T):  # fwd_finish production (rows_per_cta=4, col_split=2): ceil(T / 2) CTAs
    return row_cta_grid(T, 4, 2)


def k5_grid(T):  # bwd_prepare production (rows_per_cta=2): ceil(T / 2) CTAs
    return row_cta_grid(T, 2, 1)


def t_pad(T: int) -> int:
    """``T_pad = 128 ceil(T / 128)``, the row count of the padded records."""
    return TILE_T * (-(-T // TILE_T))


def _round256(n: int) -> int:
    return 256 * (-(-n // 256))


@functools.lru_cache(maxsize=4096)
def host_plan(rows: int, sm_count: int) -> dict[str, int]:
    """Every host-rule output of one ``(T, SM count)``: the token padding, the
    K-unit count and grid of the mapping pass, the coefficient grid, the chunk
    count, grid and rows per chunk of the column-slab pass and the token-owner
    chunking.  ``rows`` must be positive."""

    rows = int(rows)
    sm_count = int(sm_count)
    if rows <= 0:
        raise ValueError("rows must be positive")
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    p_k, units1, grid1 = k1_units(rows, sm_count)
    parts, units7, grid7, rpc = k7_chunks(rows, sm_count)
    g6, r6 = k6_grid(rows, sm_count)
    padded = t_pad(rows)
    return {
        "rows": rows,
        "sm_count": sm_count,
        "t_pad": padded,
        "m_tiles": padded // TILE_T,
        "p_k": p_k,
        "log2_pk": int(math.log2(p_k)),
        "k1_units": units1,
        "k1_grid": grid1,
        "k2_grid": k2_grid(rows),
        "k3_grid": k3_grid(rows),
        "k4_grid": k4_grid(rows),
        "k5_grid": k5_grid(rows),
        "parts": parts,
        "k7_units": units7,
        "k7_grid": grid7,
        "rows_per_chunk": rpc,
        "chunks": g6,
        "rows_per_cta": r6,
    }


_DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "uint32": 4}


def workspace_regions(kind: str, rows: int, sm_count: int) -> dict[str, Any]:
    """Byte layout of the caller-owned workspace of ``kind`` (``"fwd"`` or
    ``"bwd"``) for exactly ``rows`` tokens on a device of ``sm_count`` SMs:
    ``{"regions": {name: {"offset", "bytes", "dtype", "shape"}}, "total_bytes"}``
    with every region at a 256-byte aligned offset.  ``rows == 0`` has no
    regions (nothing is launched)."""

    if kind not in WORKSPACE_REGIONS:
        raise ValueError(
            f"unknown workspace {kind!r}; the workspaces are {sorted(WORKSPACE_REGIONS)}"
        )
    rows = int(rows)
    if rows < 0:
        raise ValueError("rows must be non-negative")
    regions: dict[str, dict[str, Any]] = {}
    if rows == 0:
        return {"regions": regions, "total_bytes": 0}
    plan = host_plan(rows, sm_count)
    offset = 0
    for region in WORKSPACE_REGIONS[kind]:
        shape = tuple(
            int(plan[dim]) if isinstance(dim, str) else int(dim)
            for dim in region["shape"]
        )
        nbytes = _DTYPE_BYTES[region["dtype"]] * math.prod(shape)
        regions[region["name"]] = {
            "offset": offset,
            "bytes": nbytes,
            "dtype": region["dtype"],
            "shape": shape,
        }
        offset += _round256(nbytes)
    return {"regions": regions, "total_bytes": offset}


def workspace_bytes(kind: str, rows: int, sm_count: int) -> int:
    """Bytes of the ``kind`` workspace for exactly ``rows`` tokens (0 for ``rows == 0``)."""

    return int(workspace_regions(kind, rows, sm_count)["total_bytes"])


@functools.lru_cache(maxsize=4096)
def workspace_bytes_upto(kind: str, rows: int, sm_count: int) -> int:
    """Bytes of a ``kind`` workspace that serves every token count ``1..rows``.

    The exact size is not monotone in the token count: the K-unit count of the
    mapping pass and the chunk count of the column-slab pass step down as the
    token count grows.  Within one token tile the unit rule can only step down
    as ``T`` grows (its cost term ``100 * P_K * T`` penalises larger ``P_K``), so
    the forward maximum over a tile is at the tile's first token count; the
    backward regions are functions of the 64-token tile count and of
    ``ceil(T / 8)`` only, non-decreasing within a tile, so its maximum over a
    tile is at the tile's last token count.  The envelope is the maximum over
    the tiles up to ``rows``; a buffer of this size therefore serves any
    ``T <= rows`` and CUDA Graph replays at smaller ``T``.
    """

    rows = int(rows)
    if rows <= 0:
        return 0
    if kind == "fwd":
        probes = [(k - 1) * TILE_T + 1 for k in range(1, -(-rows // TILE_T) + 1)]
    else:
        probes = [
            min(k * SLAB_TILE_T, rows) for k in range(1, -(-rows // SLAB_TILE_T) + 1)
        ]
    return max(workspace_bytes(kind, probe, sm_count) for probe in probes)


def rules_row(T: int, sm_count: int) -> dict[str, Any]:
    """Every host-rule output for one ``(T, sm_count)`` in the column vocabulary
    of the kernels' test references (their rules table).  ``T == 0`` is the
    no-launch row."""

    row: dict[str, Any] = {"T": T, "sm_count": sm_count, "T_pad": t_pad(T)}
    if T == 0:
        row.update(
            launch=False,
            P_K=0,
            k1_units=0,
            k1_grid=0,
            P=0,
            k7_units=0,
            k7_grid=0,
            rows_per_chunk=0,
            G6=0,
            R6=0,
        )
        row.update(
            k2_grid=0,
            row_grid4=0,
            k3_grid=0,
            k4_grid=0,
            k5_grid=0,
            fwd_workspace_bytes=0,
            bwd_workspace_bytes=0,
        )
        return row
    plan = host_plan(T, sm_count)
    row.update(
        launch=True,
        P_K=plan["p_k"],
        k1_units=plan["k1_units"],
        k1_grid=plan["k1_grid"],
        P=plan["parts"],
        k7_units=plan["k7_units"],
        k7_grid=plan["k7_grid"],
        rows_per_chunk=plan["rows_per_chunk"],
        G6=plan["chunks"],
        R6=plan["rows_per_cta"],
        k2_grid=plan["k2_grid"],
        row_grid4=row_grid(T, sm_count, 4),
        k3_grid=k3_grid(T),
        k4_grid=k4_grid(T),
        k5_grid=k5_grid(T),
        fwd_workspace_bytes=workspace_bytes("fwd", T, sm_count),
        bwd_workspace_bytes=workspace_bytes("bwd", T, sm_count),
    )
    return row


def rules_table(
    T_values: Sequence[int] = RULES_T_VALUES, sm_counts: Sequence[int] = RULES_SM_COUNTS
) -> list[dict]:
    """The rows of :func:`rules_row` for every ``T`` (ascending, duplicates
    dropped) and every SM count: the table the rule copies are checked against."""

    return [rules_row(T, sm) for T in sorted(set(T_values)) for sm in sm_counts]


# ----------------------------------------------------------------------------
# Routes


def route_key(hidden: int, kind: str) -> str:
    """Name the route of one public call: ``<kind>_h<hidden>``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return f"{kind}_h{int(hidden)}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes with every route of :data:`KINDS` on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        hidden
        for hidden in HIDDEN_SIZES
        if all(route_key(hidden, kind) in routes for kind in KINDS)
    )


def route_applies(*, device_capability: Sequence[int], hidden: int, kind: str) -> bool:
    """Whether the verified export owns one training call."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return route_key(hidden, kind) in ROUTES.get(arch, {})


def route_modules(arch: str, hidden: int, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(hidden, kind)
    names = ROUTES.get(arch, {}).get(key)
    if not names:
        raise ValueError(
            f"unsupported Cake mHC training route {key!r} on {arch!r}; exported "
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
    raise ValueError(f"unknown Cake mHC training selection rule: {kind!r}")


def selected_modules(
    arch: str, hidden: int, kind: str, *, rows: int
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``rows`` tokens.

    Each module's recorded ``select_rule`` names the row range it owns
    (``rows_between``: ``above < rows <= at_most``, ``at_most`` ``None`` =
    unbounded; ``None`` = every call); exactly the modules whose rule admits
    ``rows`` are launched, in route order.
    """

    rows = int(rows)
    names = tuple(
        name
        for name in route_modules(arch, hidden, kind)
        if _selects(MODULES[name]["select_rule"], rows)
    )
    if not names:
        raise ValueError(
            f"no exported module of the {route_key(hidden, kind)!r} route on {arch!r} admits {rows} rows"
        )
    return names


def workspace_kind(kind: str) -> str | None:
    """The workspace a route holds (``"fwd"`` / ``"bwd"``), or ``None``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return WORKSPACE_OF_KIND.get(kind)


# ----------------------------------------------------------------------------
# Launch


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``fixed`` modules launch exactly ``ctas`` CTAs (the fn split, the final
    reduction).  ``tiles`` modules launch ``ceil(rows / rows_per_tile)`` CTAs
    (the coefficient pass).  ``rows_per_cta`` modules launch
    ``ceil(rows * col_split / rows_per_cta)`` CTAs, capped at ``grid_limit``
    when positive, else at ``grid_per_sm * SM count`` when positive (the row
    streams, one (sub-)token per thread group and trip: :func:`k3_grid` =
    ``2 rows`` for the collapse, :func:`k4_grid` = ``ceil(rows / 2)`` for the
    residual mixing, :func:`k5_grid` = ``ceil(rows / 2)`` for the backward
    prepare; neither cap is set on the shipped records).  ``units_bounded`` modules launch
    ``min(ctas_per_sm * SM count, ceil(rows / tile_rows) * P_K)`` persistent
    CTAs over the (token tile, K unit) work units by static grid-stride (the
    mapping pass; ``P_K`` from :func:`k1_units`; the ticket region is bound for
    the opt-in ticket schedule and left unread by the static form).  ``chunks`` modules
    launch ``min(ctas_per_sm * SM count, ceil(rows / group_rows))`` CTAs, each
    owning a static run of token groups (the token-owner pass).  ``slab_chunks``
    modules launch ``min(SM count, slabs * P)`` CTAs over (column slab, token
    chunk) units (the column-slab pass; ``P`` from :func:`k7_chunks`).
    ``rows`` must be positive; the public API returns empty outputs for zero
    rows without launching.
    """

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "fixed":
        grid_x = int(rule["ctas"])
    elif kind == "tiles":
        grid_x = _ceil_div(rows, int(rule["rows_per_tile"]))
    elif kind == "rows_per_cta":
        grid_x = _ceil_div(
            rows * int(rule.get("col_split", 1)), int(rule["rows_per_cta"])
        )
        limit = int(rule.get("grid_limit", 0))
        per_sm = int(rule.get("grid_per_sm", 0))
        if limit <= 0 and per_sm > 0:
            limit = per_sm * sm_count(device_index)
        if limit > 0:
            grid_x = min(grid_x, limit)
    elif kind == "units_bounded":
        plan = host_plan(rows, sm_count(device_index))
        units = _ceil_div(rows, int(rule["tile_rows"])) * plan["p_k"]
        grid_x = min(int(rule["ctas_per_sm"]) * sm_count(device_index), units)
    elif kind == "chunks":
        grid_x = min(
            int(rule["ctas_per_sm"]) * sm_count(device_index),
            _ceil_div(rows, int(rule["group_rows"])),
        )
    elif kind == "slab_chunks":
        plan = host_plan(rows, sm_count(device_index))
        grid_x = min(sm_count(device_index), int(rule["slabs"]) * plan["parts"])
    else:
        raise ValueError(f"unknown Cake mHC training grid rule: {kind!r}")
    return (max(grid_x, 1), 1, 1)


def workspace_layout(kind: str, rows: int, *, device_index: int) -> dict[str, Any]:
    """:func:`workspace_regions` of ``kind`` for ``rows`` tokens on ``device_index``."""

    return workspace_regions(kind, rows, sm_count(device_index))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake mHC training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake mHC training module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    from . import cake_build

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    from . import cake_build

    return cake_build.jit_specs(arch)


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``
    to tensors and Python scalars; the module record maps its kernel parameter
    names onto that vocabulary, and a scalar the record fixes itself
    (:data:`RECORD_SCALARS`) overrides the caller's value.  The launch grid
    follows the module's recorded grid rule for ``rows`` tokens and the kernel
    runs on the current torch stream.
    """

    import tvm_ffi

    record = MODULES[name]
    logical = record["logical"]
    fixed = RECORD_SCALARS.get(name, {})
    grid = launch_grid(record, rows, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name in fixed:
            args.append(fixed[logical_name])
            continue
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
    "DZP_COLS",
    "GSTATS_COLS",
    "GSTATS_LAYOUT",
    "HC_EPS",
    "HIDDEN",
    "HIDDEN_SIZES",
    "HIDDEN_TOTAL",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAPPING_ROWS",
    "MODULES",
    "N_PAD",
    "PARAM_COLS",
    "PARAM_LAYOUT",
    "PIECES",
    "RECORD_SCALARS",
    "RMS_EPS",
    "ROUTES",
    "RULES_SM_COUNTS",
    "RULES_T_VALUES",
    "SINKHORN_ITERS",
    "SLAB_D",
    "SLAB_TILE_T",
    "STATS_COLS",
    "STATS_LAYOUT",
    "STREAMS",
    "TICKET_WORDS",
    "TILE_T",
    "TOKP_COLS",
    "TOKP_LAYOUT",
    "WORKSPACE_OF_KIND",
    "WORKSPACE_REGIONS",
    "arch_for_capability",
    "device_arch",
    "host_plan",
    "k1_units",
    "k2_grid",
    "k3_grid",
    "k4_grid",
    "k5_grid",
    "k6_grid",
    "k7_chunks",
    "launch_grid",
    "library_name",
    "load",
    "route_applies",
    "route_key",
    "route_modules",
    "row_cta_grid",
    "row_grid",
    "rules_row",
    "rules_table",
    "run_module",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "supported_hidden_sizes",
    "t_pad",
    "workspace_bytes",
    "workspace_bytes_upto",
    "workspace_kind",
    "workspace_layout",
    "workspace_regions",
]
